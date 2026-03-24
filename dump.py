import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import json
import logging

from .utils import tensor2pil, pil2tensor

logger = logging.getLogger(__name__)


class FaceBlurDetectionNode:
    """
    Face-specific blur detection that works across any resolution (100x100 to 4K+).
    
    Pipeline:
    1. Locate face region (from bounding box JSON, or full image fallback)
    2. Crop & normalize the face to a standard analysis size
    3. Run blur metrics on the face crop only
    4. Eye region gets extra weight (sharpest feature on a face)
    
    All kernel sizes, block sizes, FFT cutoffs scale dynamically
    based on the actual face crop dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_sensitivity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0=lenient (flags only severe blur), 1=strict (flags subtle softness)"
                }),
                "eye_weight": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much eye region sharpness matters vs overall face. "
                               "0.6 = eyes contribute 60% of the score"
                }),
                "uniform_region_floor": ("FLOAT", {
                    "default": 0.002, "min": 0.0, "max": 0.05, "step": 0.0005,
                    "tooltip": "Tiles below this intensity variance = uniform skin, not blur"
                }),
            },
            "optional": {
                "face_bbox_json": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "JSON with face bounding box from MoonDream/detection node. "
                               "If empty, analyzes full image as face."
                }),
                "face_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Extra padding around face bbox (fraction of face size)"
                }),
                "depth_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "specular_map": ("IMAGE",),
                "focus_distance": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "dof_falloff": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "blur_score",
        "sharpness_score",
        "face_blur_map",
        "visualization",
        "debug_info",
    )
    FUNCTION = "detect_face_blur"
    CATEGORY = "caimera_nodes/analysis"

    # Standard size we normalize face crops to before analysis.
    # Small enough to handle 100x100 inputs, large enough for detail.
    ANALYSIS_TARGET = 256

    # Eye region relative to face bbox (approximate, works for most orientations)
    # (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
    EYE_REGION = (0.20, 0.45, 0.10, 0.90)

    # Nose-mouth region (secondary sharpness check)
    NOSE_MOUTH_REGION = (0.45, 0.80, 0.20, 0.80)

    # ------------------------------------------------------------------ #
    #  Resolution-adaptive helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _adaptive_kernel_size(dim: int, fraction: float = 0.08, min_k: int = 3, max_k: int = 63) -> int:
        """
        Compute kernel size as a fraction of the image dimension.
        Always returns odd number. Clamps to [min_k, max_k].
        Works for 100px faces and 2000px faces alike.
        """
        k = int(dim * fraction)
        k = max(min_k, min(k, max_k))
        if k % 2 == 0:
            k += 1
        return k

    @staticmethod
    def _adaptive_block_sizes(h: int, w: int) -> tuple:
        """
        Generate multi-scale block sizes that make sense for the given dimensions.
        For a 100x100 face: maybe [8, 16, 32]
        For a 1000x1000 face: [32, 64, 128]
        """
        min_dim = min(h, w)

        if min_dim < 64:
            sizes = [4, 8, 16]
            weights = [0.25, 0.35, 0.40]
        elif min_dim < 150:
            sizes = [8, 16, 32]
            weights = [0.20, 0.35, 0.45]
        elif min_dim < 400:
            sizes = [16, 32, 64]
            weights = [0.20, 0.35, 0.45]
        else:
            sizes = [32, 64, 128]
            weights = [0.20, 0.35, 0.45]

        # Ensure no block is larger than half the image
        half = min_dim // 2
        sizes = [min(s, half) for s in sizes]
        sizes = [max(s, 4) for s in sizes]  # minimum 4

        return sizes, weights

    @staticmethod
    def _adaptive_freq_cutoff(h: int, w: int) -> float:
        """Scale FFT cutoff by face crop size."""
        ref = min(h, w)
        if ref < 64:
            return 0.15
        elif ref < 200:
            return 0.20
        elif ref < 500:
            return 0.28
        else:
            return 0.20 + 0.10 * (ref / 1024.0)

    # ------------------------------------------------------------------ #
    #  Face extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_face_crop(image_tensor: torch.Tensor, bbox_json: str,
                           padding: float) -> tuple:
        """
        Extract face region from image using bbox JSON.
        Returns (face_crop_tensor, bbox_coords_dict, used_full_image: bool)
        
        Handles:
        - MoonDream format: {"objects": [{"x_min":..., "y_min":..., ...}]}
        - Simple format: {"x_min":..., "y_min":..., ...}
        - Empty/invalid JSON → full image fallback
        """
        if image_tensor.dim() == 4:
            img = image_tensor[0]  # (H, W, C)
        else:
            img = image_tensor

        h, w, c = img.shape
        used_full = False
        bbox = None

        # Try to parse bbox
        if bbox_json and bbox_json.strip():
            try:
                data = json.loads(bbox_json)

                # MoonDream multi-object format
                if "objects" in data and len(data["objects"]) > 0:
                    # Use the first (or largest) face
                    best = None
                    best_area = 0
                    for obj in data["objects"]:
                        if all(k in obj for k in ("x_min", "y_min", "x_max", "y_max")):
                            area = (obj["x_max"] - obj["x_min"]) * (obj["y_max"] - obj["y_min"])
                            if area > best_area:
                                best = obj
                                best_area = area
                    if best:
                        bbox = best

                # Simple format
                elif all(k in data for k in ("x_min", "y_min", "x_max", "y_max")):
                    bbox = data

            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        if bbox is None:
            # Full image fallback
            face_crop = img
            used_full = True
            bbox = {"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}
        else:
            # Convert normalized coords to pixels and add padding
            bw = bbox["x_max"] - bbox["x_min"]
            bh = bbox["y_max"] - bbox["y_min"]

            x_min = max(0, bbox["x_min"] - bw * padding)
            y_min = max(0, bbox["y_min"] - bh * padding)
            x_max = min(1.0, bbox["x_max"] + bw * padding)
            y_max = min(1.0, bbox["y_max"] + bh * padding)

            px_x0 = int(x_min * w)
            px_y0 = int(y_min * h)
            px_x1 = int(x_max * w)
            px_y1 = int(y_max * h)

            # Guard against degenerate boxes
            if px_x1 - px_x0 < 4 or px_y1 - px_y0 < 4:
                face_crop = img
                used_full = True
            else:
                face_crop = img[px_y0:px_y1, px_x0:px_x1, :]
                used_full = False
                bbox = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}

        return face_crop, bbox, used_full

    # ------------------------------------------------------------------ #
    #  Core blur metrics (resolution-aware)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_local_contrast(gray: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Remove intensity bias. Kernel size adapts to face crop size."""
        gray_4d = gray.unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2
        local_mean = F.avg_pool2d(gray_4d, kernel_size, stride=1, padding=pad)
        local_var = F.avg_pool2d(
            (gray_4d - local_mean) ** 2,
            kernel_size, stride=1, padding=pad
        )
        local_std = torch.sqrt(local_var + 1e-6)
        return ((gray_4d - local_mean) / local_std).squeeze()

    @staticmethod
    def _laplacian_variance(gray: torch.Tensor) -> torch.Tensor:
        kernel = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32, device=gray.device
        ).unsqueeze(0).unsqueeze(0)

        g = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(1)
        return F.conv2d(g, kernel, padding=1).squeeze()

    @staticmethod
    def _gradient_energy(gray: torch.Tensor) -> torch.Tensor:
        g = gray.unsqueeze(0).unsqueeze(0) if gray.dim() == 2 else gray.unsqueeze(1)

        sx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=gray.device
        ).unsqueeze(0).unsqueeze(0)
        sy = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=gray.device
        ).unsqueeze(0).unsqueeze(0)

        gx = F.conv2d(g, sx, padding=1)
        gy = F.conv2d(g, sy, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).squeeze()

    @staticmethod
    def _frequency_energy(gray_np: np.ndarray, cutoff: float) -> float:
        """Returns high-freq energy ratio. Handles tiny images gracefully."""
        h, w = gray_np.shape
        if h < 8 or w < 8:
            # Too small for meaningful FFT
            return 0.5  # neutral

        f = np.fft.fft2(gray_np)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))

        cy, cx = h // 2, w // 2
        radius = max(2, int(min(h, w) * cutoff))

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        high_mask = dist > radius

        total = mag.sum() + 1e-8
        high = mag[high_mask].sum()
        return float(high / total)

    # ------------------------------------------------------------------ #
    #  Tiled blur map with uniform exclusion
    # ------------------------------------------------------------------ #

    def _compute_tile_map(self, gray_norm: torch.Tensor, gray_raw: torch.Tensor,
                          block_size: int, freq_cutoff: float,
                          uniform_floor: float) -> torch.Tensor:
        h, w = gray_norm.shape
        bh = max(1, h // block_size)
        bw = max(1, w // block_size)

        tiles = torch.full((bh, bw), float('nan'), dtype=torch.float32, device=gray_norm.device)

        for i in range(bh):
            for j in range(bw):
                y0, x0 = i * block_size, j * block_size
                y1, x1 = min(y0 + block_size, h), min(x0 + block_size, w)

                tile_raw = gray_raw[y0:y1, x0:x1]
                tile_norm = gray_norm[y0:y1, x0:x1]

                if tile_raw.numel() < 4:
                    continue

                # Uniform skin patch → skip, not blur
                if tile_raw.var().item() < uniform_floor:
                    continue

                lap = self._laplacian_variance(tile_norm)
                grad = self._gradient_energy(tile_norm)

                tile_np = tile_raw.cpu().numpy()
                freq = self._frequency_energy(tile_np, freq_cutoff)

                # Combined score
                score = lap.var().item() + grad.mean().item() + freq * 500
                tiles[i, j] = score

        # Fill uniform tiles with median
        valid = ~torch.isnan(tiles)
        if valid.any():
            med = tiles[valid].median()
            tiles[torch.isnan(tiles)] = med
        else:
            tiles = torch.ones_like(tiles) * 0.5

        # Normalize
        t_min, t_max = tiles.min(), tiles.max()
        if t_max - t_min > 1e-6:
            tiles = (tiles - t_min) / (t_max - t_min)
        else:
            tiles = torch.ones_like(tiles) * 0.5

        return F.interpolate(
            tiles.unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze()

    def _multiscale_face_blur_map(self, gray_norm, gray_raw, freq_cutoff, uniform_floor):
        h, w = gray_norm.shape
        sizes, weights = self._adaptive_block_sizes(h, w)

        maps = []
        for bs in sizes:
            m = self._compute_tile_map(gray_norm, gray_raw, bs, freq_cutoff, uniform_floor)
            maps.append(m)

        combined = sum(m * wt for m, wt in zip(maps, weights))
        c_min, c_max = combined.min(), combined.max()
        if c_max - c_min > 1e-6:
            combined = (combined - c_min) / (c_max - c_min)
        return combined

    # ------------------------------------------------------------------ #
    #  Eye region analysis (strongest blur indicator on a face)
    # ------------------------------------------------------------------ #

    def _analyze_face_region(self, gray_norm: torch.Tensor, gray_raw: torch.Tensor,
                             region_bounds: tuple, freq_cutoff: float) -> dict:
        """
        Analyze a sub-region of the face crop.
        region_bounds: (y_start_frac, y_end_frac, x_start_frac, x_end_frac)
        Returns dict with sharpness metrics.
        """
        h, w = gray_norm.shape
        y0 = int(h * region_bounds[0])
        y1 = int(h * region_bounds[1])
        x0 = int(w * region_bounds[2])
        x1 = int(w * region_bounds[3])

        # Guard
        if y1 - y0 < 4 or x1 - x0 < 4:
            return {"lap_var": 0.0, "grad_mean": 0.0, "freq_ratio": 0.5, "valid": False}

        region_norm = gray_norm[y0:y1, x0:x1]
        region_raw = gray_raw[y0:y1, x0:x1]

        lap = self._laplacian_variance(region_norm)
        grad = self._gradient_energy(region_norm)
        freq = self._frequency_energy(region_raw.cpu().numpy(), freq_cutoff)

        return {
            "lap_var": lap.var().item(),
            "grad_mean": grad.mean().item(),
            "freq_ratio": freq,
            "valid": True,
        }

    # ------------------------------------------------------------------ #
    #  Depth/Normal/Specular helpers (same Blinn-Phong logic, face-cropped)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _crop_map_to_face(map_tensor: torch.Tensor, bbox: dict, target_h: int, target_w: int) -> torch.Tensor:
        """Crop any auxiliary map to the face bbox region and resize."""
        if map_tensor.dim() == 4:
            m = map_tensor[0]
        else:
            m = map_tensor

        h, w = m.shape[0], m.shape[1]
        px_y0 = int(bbox["y_min"] * h)
        px_y1 = int(bbox["y_max"] * h)
        px_x0 = int(bbox["x_min"] * w)
        px_x1 = int(bbox["x_max"] * w)

        px_y0 = max(0, px_y0)
        px_y1 = min(h, max(px_y0 + 1, px_y1))
        px_x0 = max(0, px_x0)
        px_x1 = min(w, max(px_x0 + 1, px_x1))

        cropped = m[px_y0:px_y1, px_x0:px_x1]

        if cropped.dim() == 2:
            cropped = cropped.unsqueeze(0).unsqueeze(0)
        else:
            cropped = cropped.permute(2, 0, 1).unsqueeze(0)

        resized = F.interpolate(cropped, size=(target_h, target_w),
                                mode='bilinear', align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0) if resized.shape[1] > 1 else resized.squeeze()

    @staticmethod
    def _depth_focus_weight(depth_gray: torch.Tensor, focus_distance: float,
                            dof_falloff: float) -> torch.Tensor:
        if depth_gray.dim() > 2:
            depth_gray = depth_gray.mean(dim=-1) if depth_gray.shape[-1] <= 4 else depth_gray[:, :, 0]

        d_min, d_max = depth_gray.min(), depth_gray.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_gray - d_min) / (d_max - d_min)
        else:
            return torch.ones_like(depth_gray)

        proximity = 1.0 - torch.abs(depth_norm - focus_distance)
        return torch.pow(torch.clamp(proximity, 0.0, 1.0), dof_falloff)

    @staticmethod
    def _specular_sharpness(spec_gray: torch.Tensor, pool_k: int) -> torch.Tensor:
        if spec_gray.dim() > 2:
            spec_gray = spec_gray.mean(dim=-1) if spec_gray.shape[-1] <= 4 else spec_gray[:, :, 0]

        s = spec_gray.unsqueeze(0).unsqueeze(0)
        pad = pool_k // 2
        local_max = F.max_pool2d(s, kernel_size=pool_k, stride=1, padding=pad)
        local_avg = F.avg_pool2d(s, kernel_size=pool_k, stride=1, padding=pad)
        ratio = ((local_max - local_avg) / (local_avg + 1e-6)).squeeze()

        r_min, r_max = ratio.min(), ratio.max()
        if r_max - r_min > 1e-6:
            ratio = (ratio - r_min) / (r_max - r_min)
        return ratio

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_face_visualization(image: torch.Tensor, blur_map: torch.Tensor,
                                   bbox: dict, used_full: bool) -> torch.Tensor:
        """Heatmap on face region, bbox outline on full image."""
        if image.dim() == 4:
            img = image[0]
        else:
            img = image
        h_img, w_img, c = img.shape

        # Create heatmap for face region
        fh, fw = blur_map.shape
        heatmap = torch.zeros(fh, fw, 3, dtype=torch.float32, device=blur_map.device)
        blur_inv = 1.0 - blur_map
        heatmap[:, :, 0] = torch.clamp(blur_inv * 2.0, 0, 1)
        heatmap[:, :, 1] = torch.clamp(2.0 - blur_inv * 2.0, 0, 1)
        heatmap[:, :, 2] = torch.clamp(1.0 - blur_inv * 3.0, 0, 1)

        # Build full-image visualization
        pil_img = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)

        if not used_full:
            # Draw face bbox
            bx0 = int(bbox["x_min"] * w_img)
            by0 = int(bbox["y_min"] * h_img)
            bx1 = int(bbox["x_max"] * w_img)
            by1 = int(bbox["y_max"] * h_img)
            draw.rectangle([bx0, by0, bx1, by1], outline=(0, 255, 0), width=max(1, min(w_img, h_img) // 200))

            # Overlay heatmap on face region
            face_heat_pil = Image.fromarray(
                (heatmap.cpu().numpy() * 255).astype(np.uint8)
            ).resize((bx1 - bx0, by1 - by0), Image.BILINEAR)

            face_region = pil_img.crop((bx0, by0, bx1, by1))
            blended = Image.blend(face_region, face_heat_pil, alpha=0.45)
            pil_img.paste(blended, (bx0, by0))

            # Draw eye region indicator
            eye_y0 = by0 + int((by1 - by0) * 0.20)
            eye_y1 = by0 + int((by1 - by0) * 0.45)
            eye_x0 = bx0 + int((bx1 - bx0) * 0.10)
            eye_x1 = bx0 + int((bx1 - bx0) * 0.90)
            draw.rectangle([eye_x0, eye_y0, eye_x1, eye_y1],
                           outline=(255, 255, 0), width=max(1, min(w_img, h_img) // 300))
        else:
            # Full image heatmap blend
            heat_pil = Image.fromarray(
                (heatmap.cpu().numpy() * 255).astype(np.uint8)
            ).resize((w_img, h_img), Image.BILINEAR)
            pil_img = Image.blend(pil_img, heat_pil, alpha=0.45)

        result = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
        return result.unsqueeze(0)

    # ------------------------------------------------------------------ #
    #  Main entry
    # ------------------------------------------------------------------ #

    def detect_face_blur(self, image, blur_sensitivity, eye_weight, uniform_region_floor,
                         face_bbox_json="", face_padding=0.15,
                         depth_map=None, normal_map=None, specular_map=None,
                         focus_distance=0.5, dof_falloff=2.0):

        debug = []

        # --- 1. Extract face crop ---
        face_crop, bbox, used_full = self._extract_face_crop(image, face_bbox_json, face_padding)
        fh, fw = face_crop.shape[0], face_crop.shape[1]
        debug.append(f"Face crop: {fw}x{fh} ({'full image' if used_full else 'from bbox'})")

        # --- 2. Grayscale + contrast normalization ---
        if face_crop.dim() == 2:
            gray_raw = face_crop
        elif face_crop.shape[-1] >= 3:
            gray_raw = face_crop[:, :, 0] * 0.2989 + face_crop[:, :, 1] * 0.5870 + face_crop[:, :, 2] * 0.1140
        else:
            gray_raw = face_crop[:, :, 0]

        contrast_kernel = self._adaptive_kernel_size(min(fh, fw), fraction=0.08, min_k=3, max_k=63)
        gray_norm = self._normalize_local_contrast(gray_raw, contrast_kernel)
        debug.append(f"Contrast kernel: {contrast_kernel}")

        # --- 3. Adaptive parameters ---
        freq_cutoff = self._adaptive_freq_cutoff(fh, fw)
        block_sizes, block_weights = self._adaptive_block_sizes(fh, fw)
        debug.append(f"Freq cutoff: {freq_cutoff:.3f}")
        debug.append(f"Block sizes: {block_sizes}")

        # --- 4. Overall face blur map ---
        blur_map = self._multiscale_face_blur_map(gray_norm, gray_raw, freq_cutoff, uniform_region_floor)

        # --- 5. Eye region analysis (primary signal) ---
        eye_metrics = self._analyze_face_region(gray_norm, gray_raw, self.EYE_REGION, freq_cutoff)
        nose_metrics = self._analyze_face_region(gray_norm, gray_raw, self.NOSE_MOUTH_REGION, freq_cutoff)

        if eye_metrics["valid"]:
            debug.append(f"Eye: lap={eye_metrics['lap_var']:.3f} "
                         f"grad={eye_metrics['grad_mean']:.4f} "
                         f"freq={eye_metrics['freq_ratio']:.4f}")
        if nose_metrics["valid"]:
            debug.append(f"Nose/mouth: lap={nose_metrics['lap_var']:.3f} "
                         f"grad={nose_metrics['grad_mean']:.4f}")

        # --- 6. Map-based weighting (crop maps to face region) ---
        if depth_map is not None and not used_full:
            depth_face = self._crop_map_to_face(depth_map, bbox, fh, fw)
            dw = self._depth_focus_weight(depth_face, focus_distance, dof_falloff)
            if dw.dim() > 2:
                dw = dw.squeeze()
            dw = F.interpolate(dw.unsqueeze(0).unsqueeze(0),
                               size=(fh, fw), mode='bilinear', align_corners=False).squeeze()
            blur_map = blur_map * dw + blur_map * 0.3 * (1.0 - dw)
            debug.append("Depth-aware weighting applied to face")

        if specular_map is not None and not used_full:
            spec_face = self._crop_map_to_face(specular_map, bbox, fh, fw)
            pool_k = self._adaptive_kernel_size(min(fh, fw), fraction=0.06, min_k=3, max_k=31)
            if pool_k % 2 == 0:
                pool_k += 1
            ss = self._specular_sharpness(spec_face, pool_k)
            ss = F.interpolate(ss.unsqueeze(0).unsqueeze(0),
                               size=(fh, fw), mode='bilinear', align_corners=False).squeeze()
            blur_map = blur_map * (0.8 + 0.2 * ss)
            debug.append(f"Specular sharpness applied (pool_k={pool_k})")

        # Re-normalize
        bm_min, bm_max = blur_map.min(), blur_map.max()
        if bm_max - bm_min > 1e-6:
            blur_map = (blur_map - bm_min) / (bm_max - bm_min)

        # --- 7. Composite score: eye region + overall face ---
        overall_face_sharpness = blur_map.mean().item()

        if eye_metrics["valid"]:
            # Normalize eye metrics into a 0-1 sharpness score
            # These denominators are empirical anchors
            eye_lap_s = min(eye_metrics["lap_var"] / max(overall_face_sharpness * 200 + 1.0, 1.0), 1.0)
            eye_grad_s = min(eye_metrics["grad_mean"] / 0.15, 1.0)
            eye_freq_s = min(eye_metrics["freq_ratio"] / 0.5, 1.0)
            eye_sharpness = 0.35 * eye_lap_s + 0.40 * eye_grad_s + 0.25 * eye_freq_s
        else:
            eye_sharpness = overall_face_sharpness

        if nose_metrics["valid"]:
            nose_grad_s = min(nose_metrics["grad_mean"] / 0.12, 1.0)
        else:
            nose_grad_s = overall_face_sharpness

        # Weighted blend: eyes dominate, nose/mouth secondary, rest of face tertiary
        rest_weight = max(0.0, 1.0 - eye_weight - 0.15)
        sharpness = (eye_weight * eye_sharpness +
                     0.15 * nose_grad_s +
                     rest_weight * overall_face_sharpness)

        # Sensitivity curve
        sensitivity_power = 2.0 - blur_sensitivity * 1.5  # [0.5 .. 2.0]
        sharpness = float(np.clip(sharpness, 0.0, 1.0))
        blur_score = float(np.clip((1.0 - sharpness) ** sensitivity_power, 0.0, 1.0))
        sharpness_final = 1.0 - blur_score

        debug.append(f"Eye sharpness: {eye_sharpness:.4f} (weight={eye_weight})")
        debug.append(f"Overall face sharpness: {overall_face_sharpness:.4f}")
        debug.append(f"Blur score: {blur_score:.4f} | Sharpness: {sharpness_final:.4f}")

        # --- 8. Output mask (full image size, face region marked) ---
        if image.dim() == 4:
            full_h, full_w = image.shape[1], image.shape[2]
        else:
            full_h, full_w = image.shape[0], image.shape[1]

        full_mask = torch.zeros(full_h, full_w, dtype=torch.float32, device=image.device)

        if not used_full:
            # Place face blur map into full-image mask
            px_y0 = int(bbox["y_min"] * full_h)
            px_y1 = int(bbox["y_max"] * full_h)
            px_x0 = int(bbox["x_min"] * full_w)
            px_x1 = int(bbox["x_max"] * full_w)

            face_blur_resized = F.interpolate(
                (1.0 - blur_map).unsqueeze(0).unsqueeze(0),
                size=(px_y1 - px_y0, px_x1 - px_x0),
                mode='bilinear', align_corners=False
            ).squeeze()

            full_mask[px_y0:px_y1, px_x0:px_x1] = face_blur_resized
        else:
            full_mask = F.interpolate(
                (1.0 - blur_map).unsqueeze(0).unsqueeze(0),
                size=(full_h, full_w),
                mode='bilinear', align_corners=False
            ).squeeze()

        # --- 9. Visualization ---
        viz = self._create_face_visualization(image, blur_map, bbox, used_full)

        debug_info = " | ".join(debug)
        return (blur_score, sharpness_final, full_mask.unsqueeze(0), viz, debug_info)


# ------------------------------------------------------------------ #
#  Registration
# ------------------------------------------------------------------ #

NODE_CLASS_MAPPINGS = {
    "FaceBlurDetectionNode": FaceBlurDetectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceBlurDetectionNode": "Face Blur Detection (Adaptive Resolution)",
}
