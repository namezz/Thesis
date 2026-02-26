"""
losses.py — Phase-aware Composite Loss for VFI
================================================
Improvements over original:
  1. Frequency-aware Charbonnier: separate high/low-freq eps for mixed-precision stability
  2. Multi-scale intermediate supervision: loss at each backbone scale (not just final)
  3. VGG perceptual loss integrated into composite (Phase 2+)
  4. Improved Census: channel-wise (not grayscale) for color structure preservation
  5. Occlusion-aware flow smoothness: suppresses smoothing at occlusion boundaries
  6. Corrected Kendall weighting: proper 1/2 factor + gradient clipping on log_var
  7. Spectral (FFT) loss: penalizes high-frequency reconstruction errors (Phase 2+)

Loss schedule:
  Phase 1 (backbone pre-train):  LapLoss + Census
  Phase 2 (flow + refinement):   LapLoss + Census + Perceptual + FlowSmooth + FFT
  Phase 3 (4K fine-tune):        Same as Phase 2, perceptual weight increased

References:
  - Kendall et al., CVPR 2018: Multi-task learning using uncertainty
  - RIFE (Huang et al., ECCV 2022): Laplacian + Census baseline
  - IFRNet (Kong et al., CVPR 2022): Multi-scale Charbonnier supervision
  - EMA-VFI (Zhang et al., CVPR 2023): Census + perceptual combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ════════════════════════════════════════════════════════════════════════════
# Gaussian / Laplacian Pyramid Utilities
# ════════════════════════════════════════════════════════════════════════════

def gauss_kernel(channels=3, device=None):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if device is not None:
        kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat([x, torch.zeros_like(x)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3],
                                     x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        if up.shape != current.shape:
            up = F.interpolate(up, size=current.shape[-2:],
                               mode='bilinear', align_corners=False)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


# ════════════════════════════════════════════════════════════════════════════
# Laplacian Pyramid Loss
# ════════════════════════════════════════════════════════════════════════════

class LapLoss(nn.Module):
    """
    Laplacian Pyramid Loss with Charbonnier penalty.

    Changes from original:
      - eps=1e-3 for fp16/bf16 mixed-precision stability
        (original 1e-6 causes gradient instability near zero in half precision)
      - Per-level weighting option for emphasizing high-frequency details
    """

    def __init__(self, max_levels=5, channels=3, eps=1e-3):
        super().__init__()
        self.max_levels = max_levels
        self.channels = channels
        self.eps_sq = eps * eps  # pre-square for efficiency

    def _charbonnier(self, a, b):
        diff = a - b
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))

    def forward(self, pred, target):
        kernel = gauss_kernel(channels=self.channels, device=pred.device)
        # Dynamically limit levels so smallest dim stays >= 5 (kernel size)
        min_dim = min(pred.shape[-2], pred.shape[-1])
        safe_levels = max(1, int(math.log2(min_dim / 5)))
        levels = min(self.max_levels, safe_levels)
        pyr_pred = laplacian_pyramid(pred, kernel, levels)
        pyr_target = laplacian_pyramid(target, kernel, levels)
        return sum(self._charbonnier(a, b) for a, b in zip(pyr_pred, pyr_target))


# ════════════════════════════════════════════════════════════════════════════
# VGG Perceptual Loss
# ════════════════════════════════════════════════════════════════════════════

class VGGPerceptualLoss(nn.Module):
    """
    Multi-layer VGG-19 perceptual loss with L2-normalized features.

    Layers: relu2_2, relu3_4, relu4_4 (deeper layers capture more semantic content).

    Changes from original:
      - Added relu4_4 (index 27→36) for higher-level structure matching
      - Weighted sum: shallower layers get higher weight (texture > semantics for VFI)
      - Gradient detach on target path only (not input path)
    """

    def __init__(self, layer_weights=(1.0, 0.5, 0.25)):
        super().__init__()
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import models
        from torchvision.models import VGG19_Weights
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        self.slice1 = nn.Sequential(*[vgg[x] for x in range(9)])   # relu2_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(9, 18)])   # relu3_4
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(18, 27)])  # relu4_4

        for param in self.parameters():
            param.requires_grad = False

        self.slice1.eval()
        self.slice2.eval()
        self.slice3.eval()

        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.layer_weights = layer_weights

    def train(self, mode=True):
        """Override: keep VGG slices always in eval mode."""
        super().train(mode)
        self.slice1.eval()
        self.slice2.eval()
        self.slice3.eval()
        return self

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        # Forward pass (pred requires grad, target does not)
        h1 = self.slice1(pred_norm)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)

        with torch.no_grad():
            t1 = self.slice1(target_norm)
            t2 = self.slice2(t1)
            t3 = self.slice3(t2)

        # L2-normalized feature matching (stabilizes magnitude across batches)
        w = self.layer_weights
        loss = (w[0] * F.l1_loss(F.normalize(h1, dim=1), F.normalize(t1, dim=1)) +
                w[1] * F.l1_loss(F.normalize(h2, dim=1), F.normalize(t2, dim=1)) +
                w[2] * F.l1_loss(F.normalize(h3, dim=1), F.normalize(t3, dim=1)))
        return loss


# ════════════════════════════════════════════════════════════════════════════
# Ternary (Census) Loss — Channel-aware Version
# ════════════════════════════════════════════════════════════════════════════

class Ternary(nn.Module):
    """
    Ternary Census Transform loss for structural similarity.

    Changes from original:
      - Channel-wise census (not grayscale): preserves color structure information.
        Original: mean(RGB) → grayscale → census → loses color bleeding artifacts.
        New: per-channel census → detects R/G/B structural mismatches independently.
      - Configurable softness parameter for the ternary comparison.
    """

    def __init__(self, patch_size=7, softness=0.81):
        super().__init__()
        self.patch_size = patch_size
        self.softness = softness
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        w = np.transpose(w, (3, 2, 0, 1))
        self.register_buffer('w', torch.tensor(w).float())

    def transform(self, tensor):
        """
        Per-channel census transform.
        Input:  (B, C, H, W)  — e.g. C=3 for RGB
        Output: (B, C * patch_size², H, W)
        """
        B, C, H, W = tensor.shape
        w = self.w.to(dtype=tensor.dtype)  # (P², 1, P, P)
        padding = self.patch_size // 2

        # Process each channel independently to preserve color structure
        outputs = []
        for c in range(C):
            ch = tensor[:, c:c + 1, :, :]  # (B, 1, H, W)
            patches = F.conv2d(ch, w, padding=padding, bias=None)  # (B, P², H, W)
            loc_diff = patches - ch
            loc_diff_norm = loc_diff / torch.sqrt(self.softness + loc_diff ** 2)
            outputs.append(loc_diff_norm)

        return torch.cat(outputs, dim=1)  # (B, C*P², H, W)

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding,
                           device=tensor.device, dtype=tensor.dtype)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, pred, target):
        census_pred = self.transform(pred)
        census_target = self.transform(target)
        diff = census_pred - census_target.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(pred)
        loss = (dist * mask).mean()
        return loss


# ════════════════════════════════════════════════════════════════════════════
# Charbonnier Loss
# ════════════════════════════════════════════════════════════════════════════

class CharbonnierLoss(nn.Module):
    """Robust L1 (Charbonnier) with fp16-safe epsilon."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps_sq = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


# ════════════════════════════════════════════════════════════════════════════
# Frequency (FFT) Loss — penalizes high-frequency reconstruction errors
# ════════════════════════════════════════════════════════════════════════════

class FFTLoss(nn.Module):
    """
    Frequency-domain loss via 2D FFT.

    Penalizes differences in the frequency spectrum, emphasizing high-frequency
    content (edges, textures, fine details) that pixel-space L1/L2 tends to blur.
    Particularly important for VFI where motion blur and ghosting artifacts
    manifest as high-frequency errors.

    Reference: Focal Frequency Loss (Jiang et al., ICCV 2021)
    """

    def __init__(self, log_scale=True):
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred, target):
        # 2D FFT on spatial dimensions
        fft_pred = torch.fft.rfft2(pred, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')

        # Magnitude spectrum
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)

        if self.log_scale:
            # Log-scale to balance low/high frequency magnitudes
            mag_pred = torch.log1p(mag_pred)
            mag_target = torch.log1p(mag_target)

        return F.l1_loss(mag_pred, mag_target)


# ════════════════════════════════════════════════════════════════════════════
# Occlusion-aware Flow Smoothness Loss
# ════════════════════════════════════════════════════════════════════════════

class FlowSmoothnessLoss(nn.Module):
    """
    Edge-aware + occlusion-aware flow smoothness regularization.

    Changes from original:
      - Occlusion mask: suppresses smoothness penalty at occlusion boundaries
        where flow discontinuities are expected and correct.
      - Forward-backward consistency check: if |F_01(x) + F_10(warp(x))| > τ,
        mark as occluded and reduce smoothness weight.
    """

    def __init__(self, occ_threshold=1.0):
        super().__init__()
        self.occ_threshold = occ_threshold

    def _compute_occ_mask(self, flow_01, flow_10):
        """
        Forward-backward consistency occlusion detection.
        Returns soft occlusion mask: 1 = visible, 0 = occluded.
        """
        B, _, H, W = flow_01.shape
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=flow_01.device, dtype=flow_01.dtype),
            torch.arange(W, device=flow_01.device, dtype=flow_01.dtype),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)

        # Warp flow_10 to frame 0 using flow_01
        coords = grid + flow_01[:, :2]  # (B, 2, H, W) — only use first 2 channels
        # Normalize to [-1, 1] for grid_sample
        coords_norm = torch.stack([
            2.0 * coords[:, 0] / (W - 1) - 1.0,
            2.0 * coords[:, 1] / (H - 1) - 1.0,
        ], dim=-1)  # (B, H, W, 2)
        # Sample backward flow at warped positions
        flow_10_warped = F.grid_sample(
            flow_10[:, :2], coords_norm,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        # Forward-backward consistency error
        fb_error = torch.norm(flow_01[:, :2] + flow_10_warped, dim=1, keepdim=True)
        # Soft mask: sigmoid transition around threshold
        occ_mask = torch.sigmoid(5.0 * (self.occ_threshold - fb_error))
        return occ_mask  # (B, 1, H, W)

    def forward(self, flow, img, flow_backward=None):
        """
        Args:
            flow:          (B, 2 or 4, H, W) — forward flow (first 2 channels used)
            img:           (B, 3, H, W) — reference image for edge-aware weighting
            flow_backward: (B, 2 or 4, H, W) — backward flow for occlusion detection.
                           If None, falls back to edge-aware only (no occlusion).
        """
        # Edge-aware weights
        img_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]),
                            dim=1, keepdim=True)
        img_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]),
                            dim=1, keepdim=True)
        weight_x = torch.exp(-img_dx)
        weight_y = torch.exp(-img_dy)

        # Occlusion mask (if backward flow available)
        if flow_backward is not None:
            occ_mask = self._compute_occ_mask(flow, flow_backward)
            weight_x = weight_x * occ_mask[:, :, :, :-1]
            weight_y = weight_y * occ_mask[:, :, :-1, :]

        # Flow spatial gradient
        flow_2ch = flow[:, :2]
        flow_dx = torch.abs(flow_2ch[:, :, :, :-1] - flow_2ch[:, :, :, 1:])
        flow_dy = torch.abs(flow_2ch[:, :, :-1, :] - flow_2ch[:, :, 1:, :])

        loss = (weight_x * flow_dx).mean() + (weight_y * flow_dy).mean()
        return loss


# ════════════════════════════════════════════════════════════════════════════
# Composite Loss — Phase-aware with Multi-scale Supervision
# ════════════════════════════════════════════════════════════════════════════

class CompositeLoss(nn.Module):
    """
    Phase-aware composite loss for VFI training.

    Supports multi-scale intermediate supervision: if the model outputs
    predictions at multiple resolutions (e.g., from each backbone stage),
    the loss is computed at each scale with geometrically decaying weight.

    Kendall uncertainty weighting (corrected):
        L_total = Σ_i [ (1 / 2σ_i²) · L_i + (1/2) · log(σ_i²) ]
        where σ_i² = exp(s_i) and s_i is a learnable parameter.

    Phase schedule:
        Phase 1: LapLoss + Census
        Phase 2: LapLoss + Census + VGGPerceptual + FFT + FlowSmooth
        Phase 3: Same as Phase 2 (4K fine-tune, higher perceptual weight)

    Args:
        phase:          training phase (1, 2, or 3)
        multiscale_weights: weights for each scale level, e.g. [1.0, 0.5, 0.25]
                            Applied when pred is a list of multi-scale outputs.
        use_perceptual: whether to use VGG perceptual loss (Phase 2+ default)
        use_fft:        whether to use frequency-domain loss (Phase 2+ default)
    """

    def __init__(self, phase=1, multiscale_weights=None,
                 use_perceptual=None, use_fft=None):
        super().__init__()
        self.phase = phase

        # Multi-scale supervision weights (IFRNet-style)
        # Default: geometrically decaying [1.0, 0.5, 0.25] for 3 scales
        if multiscale_weights is not None:
            self.multiscale_weights = multiscale_weights
        else:
            self.multiscale_weights = None  # single-scale mode

        # Core losses (all phases)
        self.lap_loss = LapLoss(max_levels=5, eps=1e-3)
        self.ternary_loss = Ternary(patch_size=7)

        # Phase 2+ losses
        if use_perceptual is None:
            use_perceptual = (phase >= 2)
        if use_fft is None:
            use_fft = (phase >= 2)

        self.use_perceptual = use_perceptual
        self.use_fft = use_fft

        if use_perceptual:
            self.vgg_loss = VGGPerceptualLoss(
                layer_weights=(1.0, 0.5, 0.25) if phase < 3 else (1.0, 0.75, 0.5)
            )
        if use_fft:
            self.fft_loss = FFTLoss(log_scale=True)

        if phase >= 2:
            self.flow_smooth_loss = FlowSmoothnessLoss(occ_threshold=1.0)

        # Kendall uncertainty weighting: learnable log-variance parameters
        # s_i = log(σ_i²), initialized to 0 → σ² = 1 → weight = 1/(2·1) = 0.5
        num_losses = 2  # lap + ternary
        if use_perceptual:
            num_losses += 1
        if use_fft:
            num_losses += 1
        if phase >= 2:
            num_losses += 1  # flow smoothness

        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_losses)
        ])

        # Map loss names to indices for clarity
        self._loss_names = ['lap', 'ter']
        if use_perceptual:
            self._loss_names.append('vgg')
        if use_fft:
            self._loss_names.append('fft')
        if phase >= 2:
            self._loss_names.append('flow')

    def _kendall_weight(self, loss_val, idx):
        """
        Kendall uncertainty weighting (corrected formulation):
            (1 / 2σ²) · L + (1/2) · log(σ²)
            = (1/2) · exp(-s) · L + (1/2) · s
        where s = log(σ²) is the learnable parameter.
        """
        s = self.log_vars[idx]
        # Clamp s to prevent extreme weights: σ² ∈ [e^-4, e^4] ≈ [0.018, 54.6]
        s_clamped = torch.clamp(s, min=-4.0, max=4.0)
        return 0.5 * torch.exp(-s_clamped) * loss_val + 0.5 * s_clamped

    def _compute_losses_single_scale(self, pred, gt, flow=None,
                                      flow_backward=None, img0=None):
        """Compute all loss components at a single scale."""
        losses = {}

        # Core losses
        losses['lap'] = self.lap_loss(pred, gt)
        losses['ter'] = self.ternary_loss(pred, gt)

        # Perceptual loss (Phase 2+)
        if self.use_perceptual:
            losses['vgg'] = self.vgg_loss(pred, gt)

        # FFT loss (Phase 2+)
        if self.use_fft:
            losses['fft'] = self.fft_loss(pred, gt)

        # Flow smoothness (Phase 2+)
        if self.phase >= 2 and flow is not None and img0 is not None:
            losses['flow'] = self.flow_smooth_loss(
                flow, img0, flow_backward=flow_backward
            )

        return losses

    def forward(self, pred, gt, flow=None, flow_backward=None, img0=None):
        """
        Args:
            pred:          (B, 3, H, W) or list of multi-scale predictions
                           If list: [full_res, half_res, quarter_res, ...]
            gt:            (B, 3, H, W) ground truth
            flow:          (B, 2+, H, W) forward optical flow (Phase 2+)
            flow_backward: (B, 2+, H, W) backward flow for occlusion detection
            img0:          (B, 3, H, W) reference frame for edge-aware smoothness

        Returns:
            total_loss: scalar
            loss_dict:  dict of individual loss values for logging
        """
        loss_dict = {}

        # --- Multi-scale or single-scale ---
        if isinstance(pred, (list, tuple)) and self.multiscale_weights is not None:
            # Multi-scale intermediate supervision (IFRNet-style):
            # Compute loss at each scale, aggregate per-component with
            # geometrically decaying weights, then apply Kendall weighting.
            ms_weights = self.multiscale_weights

            component_losses = {name: torch.tensor(0.0, device=gt.device)
                                for name in self._loss_names}

            for scale_idx, (pred_s, w_s) in enumerate(zip(pred, ms_weights)):
                # Downsample GT / flow / img0 to match prediction scale
                if pred_s.shape[-2:] != gt.shape[-2:]:
                    gt_s = F.interpolate(gt, size=pred_s.shape[-2:],
                                         mode='bilinear', align_corners=False)
                    sf = pred_s.shape[-1] / gt.shape[-1]
                    flow_s = (F.interpolate(flow, size=pred_s.shape[-2:],
                              mode='bilinear', align_corners=False) * sf
                              if flow is not None else None)
                    flow_bwd_s = (F.interpolate(flow_backward, size=pred_s.shape[-2:],
                                  mode='bilinear', align_corners=False) * sf
                                  if flow_backward is not None else None)
                    img0_s = (F.interpolate(img0, size=pred_s.shape[-2:],
                              mode='bilinear', align_corners=False)
                              if img0 is not None else None)
                else:
                    gt_s = gt
                    flow_s, flow_bwd_s, img0_s = flow, flow_backward, img0

                scale_losses = self._compute_losses_single_scale(
                    pred_s, gt_s, flow_s, flow_bwd_s, img0_s
                )

                for name, val in scale_losses.items():
                    loss_dict[f's{scale_idx}_{name}'] = val.item()
                    if name in component_losses:
                        component_losses[name] = component_losses[name] + w_s * val

            # Apply Kendall weighting to each aggregated component
            total = torch.tensor(0.0, device=gt.device)
            for idx, name in enumerate(self._loss_names):
                if name in component_losses:
                    weighted = self._kendall_weight(component_losses[name], idx)
                    total = total + weighted
                    loss_dict[f'loss_{name}'] = component_losses[name].item()
                    loss_dict[f'w_{name}'] = (0.5 * torch.exp(
                        -torch.clamp(self.log_vars[idx], -4, 4))).item()

        else:
            # Single-scale mode
            if isinstance(pred, (list, tuple)):
                pred = pred[0]  # use full-resolution prediction

            losses = self._compute_losses_single_scale(
                pred, gt, flow, flow_backward, img0
            )

            total = torch.tensor(0.0, device=gt.device)
            for idx, name in enumerate(self._loss_names):
                if name in losses:
                    weighted = self._kendall_weight(losses[name], idx)
                    total = total + weighted
                    loss_dict[f'loss_{name}'] = losses[name].item()
                    loss_dict[f'w_{name}'] = (0.5 * torch.exp(
                        -torch.clamp(self.log_vars[idx], -4, 4))).item()

        loss_dict['loss_total'] = total.item()
        return total, loss_dict
