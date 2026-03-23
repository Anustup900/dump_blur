import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Assuming these utils exist in your project
from .utils import tensor2pil, pil2tensor


class BlurDetectionNode:
    """
    Detects blur amount in an image using multiple methods:
    1. Laplacian variance (global sharpness)
    2. Frequency-domain energy (FFT high-freq ratio)
    3. Depth-aware blur map (Blinn-Phong inspired focus estimation)
    
    Uses normal/depth/specular maps to distinguish intentional DoF blur
    from defocus/motion blur, similar to how CromptonRelight uses maps
    for lighting — but here we use them for focus-plane estimation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": ((
                    "laplacian",
                    "frequency",
                    "gradient_energy",
                    "combined",
                ), {"default": "combined"}),
                "block_size": ("INT", {
                    "default": 32, "min": 8, "max": 256, "step": 8,
                    "tooltip": "Tile size for local blur map computation"
                }),
                "blur_threshold": ("FLOAT", {
                    "default": 100.0, "min": 0.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Laplacian variance below this = blurry"
                }),
                "frequency_cutoff": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "FFT high-frequency ratio threshold"
                }),
            },
            "optional": {
                "depth_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "specular_map": ("IMAGE",),
                "focus_distance": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Normalized depth at which the image should be in focus (0=near, 1=far)"
                }),
                "dof_falloff": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Blinn-Phong-style exponent controlling DoF sharpness falloff"
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "blur_score",        # 0-1 global blur amount (1 = very blurry)
        "sharpness_score",   # 0-1 global sharpness (1 = very sharp)
        "blur_map",          # per-pixel blur intensity mask
        "visualization",     # heatmap overlay on original
        "debug_info",
    )
    FUNCTION = "detect_blur"
    CATEGORY = "caimera_nodes/analysis"

    # ------------------------------------------------------------------ #
    #  Core blur metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def _laplacian_variance(gray: torch.Tensor) -> torch.Tensor:
        """Laplacian variance — classic sharpness measure."""
        kernel = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32, device=gray.device
        ).unsqueeze(0).unsqueeze(0)

        if gray.dim() == 2:
            gray = gray.unsqueeze(0).unsqueeze(0)
        elif gray.dim() == 3:
            gray = gray.unsqueeze(1)

        lap = F.conv2d(gray, kernel, padding=1)
        return lap.squeeze()

    @staticmethod
    def _frequency_energy(gray_np: np.ndarray, cutoff_ratio: float) -> tuple:
        """FFT-based blur detection. Returns (high_freq_ratio, magnitude_spectrum)."""
        f = np.fft.fft2(gray_np)
        fshift = np.fft.fftshift(f)
        mag = np.log1p(np.abs(fshift))

        h, w = gray_np.shape
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * cutoff_ratio)

        # Mask out low frequencies
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        high_freq_mask = dist > radius

        total_energy = mag.sum() + 1e-8
        high_energy = mag[high_freq_mask].sum()
        ratio = high_energy / total_energy

        return float(ratio), mag

    @staticmethod
    def _gradient_energy(gray: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude — high values = sharp edges."""
        if gray.dim() == 2:
            gray = gray.unsqueeze(0).unsqueeze(0)
        elif gray.dim() == 3:
            gray = gray.unsqueeze(1)

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32, device=gray.device
        ).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32, device=gray.device
        ).unsqueeze(0).unsqueeze(0)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return magnitude.squeeze()

    # ------------------------------------------------------------------ #
    #  Depth-aware focus estimation (Blinn-Phong inspired)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _depth_focus_weight(depth_map: torch.Tensor,
                            focus_distance: float,
                            dof_falloff: float) -> torch.Tensor:
        """
        Blinn-Phong-style focus weight based on depth.
        
        Analogous to specular highlight computation:
          - focus_distance acts like the "light direction" (ideal focus plane)
          - dof_falloff acts like specular_power (controls sharpness of focus region)
          - The dot product is replaced by depth proximity
        
        Returns a per-pixel weight: 1.0 = in focus plane, 0.0 = far from focus.
        """
        if depth_map.dim() == 4:
            depth = depth_map[:, :, :, 0]  # take first channel
        elif depth_map.dim() == 3:
            depth = depth_map[0] if depth_map.shape[0] <= 4 else depth_map[:, :, 0]
        else:
            depth = depth_map

        # Normalize depth to 0-1
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth - d_min) / (d_max - d_min)
        else:
            depth_norm = torch.zeros_like(depth)

        # Proximity to focus plane — Blinn-Phong style power falloff
        proximity = 1.0 - torch.abs(depth_norm - focus_distance)
        proximity = torch.clamp(proximity, 0.0, 1.0)

        # Apply specular-power-style exponent (sharper falloff = narrower DoF)
        focus_weight = torch.pow(proximity, dof_falloff)

        return focus_weight

    @staticmethod
    def _normal_variance_map(normal_map: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        High normal variance in a local region suggests geometric detail,
        so blur there is more likely defocus/motion blur (not flat surface).
        """
        if normal_map.dim() == 4:
            normals = normal_map[0]  # (H, W, 3)
        else:
            normals = normal_map

        normals = normals * 2.0 - 1.0  # to [-1, 1]
        h, w, _ = normals.shape

        # Compute local variance of normals in blocks
        var_map = torch.zeros(h, w, dtype=torch.float32, device=normals.device)
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size

        padded = F.pad(normals.permute(2, 0, 1).unsqueeze(0),
                       (0, pad_w, 0, pad_h), mode='reflect')
        padded = padded.squeeze(0)  # (3, H_pad, W_pad)

        ph, pw = padded.shape[1], padded.shape[2]
        blocks = padded.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
        # (3, num_blocks_h, num_blocks_w, block_size, block_size)

        block_var = blocks.var(dim=(-1, -2)).mean(dim=0)  # (num_bh, num_bw)

        # Upsample back
        var_map = F.interpolate(
            block_var.unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze()

        # Normalize
        v_min, v_max = var_map.min(), var_map.max()
        if v_max - v_min > 1e-6:
            var_map = (var_map - v_min) / (v_max - v_min)

        return var_map

    @staticmethod
    def _specular_sharpness_indicator(specular_map: torch.Tensor) -> torch.Tensor:
        """
        Specular highlights are point-like — if they appear soft/spread,
        the image is likely blurred. Measure local peak-to-mean ratio.
        """
        if specular_map.dim() == 4:
            spec = specular_map[0].mean(dim=-1)  # grayscale
        elif specular_map.dim() == 3:
            spec = specular_map.mean(dim=-1) if specular_map.shape[-1] <= 4 else specular_map[0]
        else:
            spec = specular_map

        # Local max via max-pool vs local mean via avg-pool
        spec_4d = spec.unsqueeze(0).unsqueeze(0)
        local_max = F.max_pool2d(spec_4d, kernel_size=15, stride=1, padding=7)
        local_avg = F.avg_pool2d(spec_4d, kernel_size=15, stride=1, padding=7)

        # High ratio = sharp specular highlights = sharp image
        ratio = (local_max - local_avg) / (local_avg + 1e-6)
        ratio = ratio.squeeze()

        # Normalize
        r_min, r_max = ratio.min(), ratio.max()
        if r_max - r_min > 1e-6:
            ratio = (ratio - r_min) / (r_max - r_min)

        return ratio

    # ------------------------------------------------------------------ #
    #  Local blur map (tiled computation)
    # ------------------------------------------------------------------ #

    def _compute_local_blur_map(self, gray: torch.Tensor, block_size: int,
                                method: str, freq_cutoff: float) -> torch.Tensor:
        """Compute per-tile blur score and upsample to full resolution."""
        h, w = gray.shape
        bh = max(1, h // block_size)
        bw = max(1, w // block_size)

        blur_tiles = torch.zeros(bh, bw, dtype=torch.float32, device=gray.device)

        for i in range(bh):
            for j in range(bw):
                y0 = i * block_size
                x0 = j * block_size
                y1 = min(y0 + block_size, h)
                x1 = min(x0 + block_size, w)
                tile = gray[y0:y1, x0:x1]

                if tile.numel() < 4:
                    continue

                if method in ("laplacian", "combined"):
                    lap = self._laplacian_variance(tile)
                    score = lap.var().item()
                elif method == "gradient_energy":
                    grad = self._gradient_energy(tile)
                    score = grad.mean().item()
                elif method == "frequency":
                    tile_np = tile.cpu().numpy()
                    ratio, _ = self._frequency_energy(tile_np, freq_cutoff)
                    score = ratio * 1000  # scale to comparable range
                else:
                    score = 0.0

                blur_tiles[i, j] = score

        # Normalize tiles
        t_min, t_max = blur_tiles.min(), blur_tiles.max()
        if t_max - t_min > 1e-6:
            blur_tiles = (blur_tiles - t_min) / (t_max - t_min)

        # Upsample to full resolution
        blur_map = F.interpolate(
            blur_tiles.unsqueeze(0).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze()

        return blur_map

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_heatmap_overlay(image: torch.Tensor, blur_map: torch.Tensor,
                                alpha: float = 0.5) -> torch.Tensor:
        """Overlay a blue-to-red heatmap of blur intensity on the image."""
        h, w = blur_map.shape
        heatmap = torch.zeros(h, w, 3, dtype=torch.float32, device=blur_map.device)

        # Blue (sharp) -> Yellow (moderate) -> Red (blurry)
        # Invert: high blur_map = sharp, so we invert for "blur intensity"
        blur_intensity = 1.0 - blur_map

        heatmap[:, :, 0] = torch.clamp(blur_intensity * 2.0, 0, 1)           # Red
        heatmap[:, :, 1] = torch.clamp(2.0 - blur_intensity * 2.0, 0, 1)     # Green (yellow band)
        heatmap[:, :, 2] = torch.clamp(1.0 - blur_intensity * 3.0, 0, 1)     # Blue

        # Resize image if needed
        if image.dim() == 4:
            img = image[0]
        else:
            img = image
        img = F.interpolate(
            img.permute(2, 0, 1).unsqueeze(0),
            size=(h, w), mode='bilinear', align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        overlay = img * (1 - alpha) + heatmap * alpha
        overlay = torch.clamp(overlay, 0, 1)
        return overlay.unsqueeze(0)

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def detect_blur(self, image, method, block_size, blur_threshold, frequency_cutoff,
                    depth_map=None, normal_map=None, specular_map=None,
                    focus_distance=0.5, dof_falloff=2.0):

        # Convert to grayscale
        if image.dim() == 4:
            img = image[0]  # (H, W, C)
        else:
            img = image

        gray = img[:, :, 0] * 0.2989 + img[:, :, 1] * 0.5870 + img[:, :, 2] * 0.1140
        h, w = gray.shape
        debug_parts = []

        # --- 1. Global Laplacian variance ---
        lap = self._laplacian_variance(gray)
        lap_var = lap.var().item()
        debug_parts.append(f"Laplacian variance: {lap_var:.2f}")

        # --- 2. Frequency energy ---
        gray_np = gray.cpu().numpy()
        freq_ratio, _ = self._frequency_energy(gray_np, frequency_cutoff)
        debug_parts.append(f"High-freq energy ratio: {freq_ratio:.4f}")

        # --- 3. Gradient energy ---
        grad = self._gradient_energy(gray)
        grad_mean = grad.mean().item()
        debug_parts.append(f"Gradient energy mean: {grad_mean:.4f}")

        # --- 4. Local blur map ---
        blur_map = self._compute_local_blur_map(gray, block_size, method, frequency_cutoff)

        # --- 5. Depth-aware weighting (Blinn-Phong style) ---
        depth_weight = None
        if depth_map is not None:
            depth_weight = self._depth_focus_weight(depth_map, focus_distance, dof_falloff)
            depth_weight = F.interpolate(
                depth_weight.unsqueeze(0).unsqueeze(0) if depth_weight.dim() == 2
                else depth_weight.unsqueeze(0),
                size=(h, w), mode='bilinear', align_corners=False
            ).squeeze()
            debug_parts.append(f"Depth-aware DoF applied (focus={focus_distance:.2f}, falloff={dof_falloff:.1f})")

            # Weight the blur map: blur in out-of-focus regions is expected,
            # blur in in-focus regions is problematic
            # depth_weight high = in focus → blur there matters more
            blur_map = blur_map * depth_weight + blur_map * 0.3 * (1 - depth_weight)

        # --- 6. Normal map detail weighting ---
        if normal_map is not None:
            normal_var = self._normal_variance_map(normal_map, block_size)
            normal_var = F.interpolate(
                normal_var.unsqueeze(0).unsqueeze(0),
                size=(h, w), mode='bilinear', align_corners=False
            ).squeeze()
            # High normal variance + low sharpness = problematic blur
            blur_map = blur_map * (0.7 + 0.3 * normal_var)
            debug_parts.append("Normal-variance weighting applied")

        # --- 7. Specular highlight sharpness ---
        if specular_map is not None:
            spec_sharp = self._specular_sharpness_indicator(specular_map)
            spec_sharp = F.interpolate(
                spec_sharp.unsqueeze(0).unsqueeze(0),
                size=(h, w), mode='bilinear', align_corners=False
            ).squeeze()
            spec_score = spec_sharp.mean().item()
            debug_parts.append(f"Specular sharpness indicator: {spec_score:.4f}")
            # Blend specular info into blur map
            blur_map = blur_map * (0.8 + 0.2 * spec_sharp)

        # Re-normalize blur_map
        bm_min, bm_max = blur_map.min(), blur_map.max()
        if bm_max - bm_min > 1e-6:
            blur_map = (blur_map - bm_min) / (bm_max - bm_min)

        # --- Compute global scores ---
        if method == "combined":
            # Weighted combination of all signals
            lap_score = min(lap_var / max(blur_threshold, 1e-6), 1.0)
            freq_score = min(freq_ratio / 0.5, 1.0)
            grad_score = min(grad_mean / 0.15, 1.0)

            sharpness = 0.45 * lap_score + 0.30 * freq_score + 0.25 * grad_score
        elif method == "laplacian":
            sharpness = min(lap_var / max(blur_threshold, 1e-6), 1.0)
        elif method == "frequency":
            sharpness = min(freq_ratio / 0.5, 1.0)
        elif method == "gradient_energy":
            sharpness = min(grad_mean / 0.15, 1.0)
        else:
            sharpness = 0.0

        sharpness = float(np.clip(sharpness, 0.0, 1.0))
        blur_score = 1.0 - sharpness

        debug_parts.append(f"Global blur score: {blur_score:.4f}")
        debug_parts.append(f"Global sharpness:  {sharpness:.4f}")
        debug_parts.append(f"Method: {method}")

        # --- Visualization ---
        viz = self._create_heatmap_overlay(image, blur_map, alpha=0.45)

        # blur_map as MASK (inverted: 1 = blurry)
        mask_out = (1.0 - blur_map).unsqueeze(0)

        debug_info = " | ".join(debug_parts)

        return (blur_score, sharpness, mask_out, viz, debug_info)


# ------------------------------------------------------------------ #
#  Registration
# ------------------------------------------------------------------ #

NODE_CLASS_MAPPINGS = {
    "BlurDetectionNode": BlurDetectionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlurDetectionNode": "Blur Detection (Blinn-Phong + Crompton)",
}
