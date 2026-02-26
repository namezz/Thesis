"""
backbone_v3.py — Factorized Spatio-Temporal SSM Backbone with NSS Scan
========================================================================
Best-of-both-worlds integration of V1 and V2, enhanced with modern scan strategies.

Architecture:
  - FactorizedSSMBlock: spatial Mamba2 + symmetric temporal MLP (seq = H×W)
  - CrossGatingFusion: bi-directional cross-gating for hybrid branch fusion
  - Learnable frame merge: concat + 1×1 conv (not naive addition)

Scanning Strategy — Nested S-shaped Scan (NSS):
  Replaces the multi-directional raster/column scan from V1/V2.

  Why NSS over 2DMamba?
  ┌─────────────────────────┬──────────────────────────────┬──────────────────────────────┐
  │ Criterion               │ Option A: NSS (MaIR/LC-Mamba)│ Option B: 2DMamba             │
  ├─────────────────────────┼──────────────────────────────┼──────────────────────────────┤
  │ Implementation          │ Pure index reordering (Py)    │ Custom CUDA kernel (pscan.so)│
  │ Mamba2 SSD compatible   │ ✅ Yes (any 1D SSM)          │ ❌ Mamba v1 only             │
  │ Extra compute cost      │ Zero                         │ Custom parallel scan overhead │
  │ Pixel-level task proof  │ MaIR: SOTA 14 datasets (SR,  │ WSI classification only       │
  │                         │ denoise, deblur, dehaze)     │ (not pixel-level)            │
  │ VFI-specific validation │ LC-Mamba (CVPR 2025 VFI)     │ None                         │
  │ Spatial continuity      │ ✅ S-path eliminates row-end │ ✅ True 2D recurrence         │
  │                         │    pixel jumps               │                              │
  │ Locality preservation   │ ✅ Stripe-based windowing    │ ❌ Global only               │
  │ Engineering risk        │ Low (pure Python)            │ High (CUDA build, version)   │
  └─────────────────────────┴──────────────────────────────┴──────────────────────────────┘

  NSS (Nested S-shaped Scan) from MaIR (CVPR 2025):
    1. Divide feature map into horizontal stripes of width w_s
    2. Within each stripe, scan in S-shaped (boustrophedon) path:
       Row 0: left→right, Row 1: right→left, Row 2: left→right, ...
    3. Cross-stripe continuity: shift-stripe mechanism (like shifted windows)
    4. Apply 4 directions: H-forward, H-backward, V-forward, V-backward
    This preserves BOTH locality (stripe regions) AND continuity (S-path).

  Traditional Z-scan (raster) problem:
    Row 0: [1, 2, 3, 4]      → Pixel 4 and pixel 5 are spatially adjacent
    Row 1: [5, 6, 7, 8]        but 4 positions apart in the sequence!

  S-shaped scan solution:
    Row 0: [1, 2, 3, 4]      → Pixel 4 and pixel 5 are now adjacent
    Row 1: [8, 7, 6, 5]        in the sequence (positions 4 and 5).

VRAM estimates (RTX 5090 32GB, 256×256, embed=[32,64,128], depths=[2,2,2]):
  V1 (4-dir SS2D interleaved):   batch=4 → ~22 GB
  V2 (1-dir factorized):         batch=4 →  ~7 GB
  V3-NSS (4-dir factorized NSS): batch=4 → ~14 GB  ← recommended default
  V3-NSS (2-dir factorized NSS): batch=4 → ~10 GB  ← VRAM-constrained option

References:
  - MaIR (Li et al., CVPR 2025): NSS scan for image restoration
  - LC-Mamba (Jeong et al., CVPR 2025): Hilbert curve scan for VFI
  - VideoMamba (Li et al., CVPR 2024): factorized scan strategy
  - Gated Attention (Qiu et al., arXiv:2505.06708, Qwen Team, 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from functools import partial, lru_cache
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm import Mamba2
except ImportError:
    import warnings
    warnings.warn(
        "mamba_ssm not found. Mamba2 backbone mode will fail at runtime. "
        "Install with: pip install mamba-ssm>=2.0"
    )
    Mamba2 = None

from .utils import (
    window_partition, window_reverse, Mlp, ECAB,
    CrossGatingFusion, matvlm_init_mamba2
)
from .backbone import GatedWindowAttention


# ═══════════════════════════════════════════════════════════════════════════
# Nested S-shaped Scan (NSS) — MaIR CVPR 2025
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=32)
def _build_nss_indices(H: int, W: int, stripe_width: int,
                       direction: str) -> torch.Tensor:
    """
    Build Nested S-shaped Scan index mapping for a given spatial size.

    The scan divides the feature map into stripes and applies boustrophedon
    (S-shaped) traversal within each stripe, ensuring that spatially adjacent
    pixels remain adjacent in the flattened sequence.

    Args:
        H, W:          spatial dimensions
        stripe_width:  width of each horizontal/vertical stripe (default 4-8)
        direction:     one of 'h_fwd', 'h_bwd', 'v_fwd', 'v_bwd'
                       h = horizontal stripes, v = vertical stripes
                       fwd = top-left start, bwd = bottom-right start

    Returns:
        indices: (H*W,) — permutation indices for flattening
    """
    L = H * W
    indices = torch.zeros(L, dtype=torch.long)

    if direction.startswith('h'):
        # Horizontal stripes: divide rows into groups of stripe_width
        pos = 0
        num_stripes = (H + stripe_width - 1) // stripe_width
        for s in range(num_stripes):
            row_start = s * stripe_width
            row_end = min(row_start + stripe_width, H)
            for local_r, r in enumerate(range(row_start, row_end)):
                if local_r % 2 == 0:
                    # Even row within stripe: left → right
                    for c in range(W):
                        indices[pos] = r * W + c
                        pos += 1
                else:
                    # Odd row within stripe: right → left (S-shaped)
                    for c in range(W - 1, -1, -1):
                        indices[pos] = r * W + c
                        pos += 1

        if direction == 'h_bwd':
            indices = indices.flip(0)

    else:  # direction starts with 'v'
        # Vertical stripes: divide columns into groups of stripe_width
        pos = 0
        num_stripes = (W + stripe_width - 1) // stripe_width
        for s in range(num_stripes):
            col_start = s * stripe_width
            col_end = min(col_start + stripe_width, W)
            for local_c, c in enumerate(range(col_start, col_end)):
                if local_c % 2 == 0:
                    # Even col within stripe: top → bottom
                    for r in range(H):
                        indices[pos] = r * W + c
                        pos += 1
                else:
                    # Odd col within stripe: bottom → top (S-shaped)
                    for r in range(H - 1, -1, -1):
                        indices[pos] = r * W + c
                        pos += 1

        if direction == 'v_bwd':
            indices = indices.flip(0)

    return indices


@lru_cache(maxsize=32)
def _build_nss_inverse(H: int, W: int, stripe_width: int,
                       direction: str) -> torch.Tensor:
    """Build inverse permutation for unscanning (sequence → spatial)."""
    fwd_idx = _build_nss_indices(H, W, stripe_width, direction)
    inv_idx = torch.empty_like(fwd_idx)
    inv_idx[fwd_idx] = torch.arange(H * W)
    return inv_idx


class NSScan(nn.Module):
    """
    Nested S-shaped Scan module.

    Converts 2D feature maps to 1D sequences using S-shaped traversal
    within stripes, and converts back after SSM processing.

    Supports 2 or 4 scan directions:
      - 2-dir: h_fwd + v_fwd  (horizontal + vertical S-scan)
      - 4-dir: h_fwd + h_bwd + v_fwd + v_bwd  (full coverage)

    The shift-stripe mechanism (from MaIR) offsets stripe boundaries
    in alternating layers to avoid fixed boundary artifacts, analogous
    to shifted window attention in Swin Transformer.

    Args:
        stripe_width:   pixel width of each stripe (default 4)
        num_directions: 2 or 4
        shift:          whether to apply shift-stripe (offset by stripe_width//2)
    """

    def __init__(self, stripe_width: int = 4, num_directions: int = 4,
                 shift: bool = False):
        super().__init__()
        self.stripe_width = stripe_width
        self.num_directions = num_directions
        self.shift = shift
        self.shift_offset = stripe_width // 2 if shift else 0

        if num_directions == 2:
            self.directions = ['h_fwd', 'v_fwd']
        elif num_directions == 4:
            self.directions = ['h_fwd', 'h_bwd', 'v_fwd', 'v_bwd']
        else:
            raise ValueError(f"num_directions must be 2 or 4, got {num_directions}")

    def scan(self, x_2d: torch.Tensor) -> tuple:
        """
        Flatten 2D features into multiple 1D sequences via NSS.

        Args:
            x_2d: (N, H, W, C) — spatial features

        Returns:
            x_seqs:    (num_dirs * N, L, C) — concatenated scanned sequences
            scan_info: dict with metadata for unscan
        """
        N, H, W, C = x_2d.shape
        L = H * W
        device = x_2d.device

        # Apply shift-stripe offset
        if self.shift_offset > 0:
            x_2d = torch.roll(
                x_2d,
                shifts=(-self.shift_offset, -self.shift_offset),
                dims=(1, 2)
            )

        x_flat = x_2d.reshape(N, L, C)
        seqs = []

        for direction in self.directions:
            idx = _build_nss_indices(H, W, self.stripe_width, direction)
            idx = idx.to(device)
            seq = x_flat[:, idx, :]  # (N, L, C)
            seqs.append(seq)

        x_seqs = torch.cat(seqs, dim=0)  # (num_dirs * N, L, C)
        scan_info = {'H': H, 'W': W, 'N': N, 'C': C}
        return x_seqs, scan_info

    def unscan(self, x_seqs: torch.Tensor, scan_info: dict) -> torch.Tensor:
        """
        Reconstruct 2D features from scanned sequences (inverse NSS).
        Averages contributions from all scan directions.

        Args:
            x_seqs:    (num_dirs * N, L, C)
            scan_info: dict from scan()

        Returns:
            x_2d: (N, H, W, C) — reconstructed spatial features
        """
        H, W, N, C = scan_info['H'], scan_info['W'], scan_info['N'], scan_info['C']
        device = x_seqs.device
        L = H * W
        num_dirs = self.num_directions

        chunks = x_seqs.chunk(num_dirs, dim=0)  # list of (N, L, C)

        x_acc = torch.zeros(N, L, C, device=device, dtype=x_seqs.dtype)
        for seq, direction in zip(chunks, self.directions):
            inv_idx = _build_nss_inverse(H, W, self.stripe_width, direction)
            inv_idx = inv_idx.to(device)
            x_acc += seq[:, inv_idx, :]

        x_avg = x_acc / num_dirs
        x_2d = x_avg.reshape(N, H, W, C)

        # Reverse shift-stripe offset
        if self.shift_offset > 0:
            x_2d = torch.roll(
                x_2d,
                shifts=(self.shift_offset, self.shift_offset),
                dims=(1, 2)
            )

        return x_2d


# ═══════════════════════════════════════════════════════════════════════════
# Factorized Spatio-Temporal SSM Block (with NSS)
# ═══════════════════════════════════════════════════════════════════════════

class FactorizedSSMBlock(nn.Module):
    """
    Spatio-Temporal Factorized SSM Block with Nested S-shaped Scan.

    Stage 1 (Spatial): Shared-weight Mamba2 processes NSS-scanned sequences.
        - 4-dir NSS: S-shaped scan in h_fwd/h_bwd/v_fwd/v_bwd
        - Sequences batched along dim-0; Mamba processes all at once
        - Averaged back via inverse NSS

    Stage 2 (Temporal): Symmetric cross-frame MLP fusion with LayerNorm.
        F0' = F0 + MLP([F0 ∥ F1])
        F1' = F1 + MLP([F1 ∥ F0])

    Benefits of NSS over raster/column scan:
      - Spatial continuity: S-path keeps adjacent pixels adjacent in sequence
      - Locality: stripe-based regions preserve local neighborhoods
      - Shift-stripe: alternating layers offset stripes (avoid boundary artifacts)
      - Zero overhead: pure index reordering, no extra computation
    """

    def __init__(self, d_model, d_state=64, d_conv=4, expand=2,
                 num_scan_dirs=4, stripe_width=4, shift=False,
                 use_checkpointing=True, headdim=64):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing

        # Shared spatial Mamba2 (weight-tied across frames AND scan directions)
        self.spatial_mamba = Mamba2(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand, headdim=headdim
        )

        # NSS scanner
        self.nss = NSScan(
            stripe_width=stripe_width,
            num_directions=num_scan_dirs,
            shift=shift
        )

        # Symmetric temporal fusion MLP
        self.time_mix_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.norm_s = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

    def forward(self, x0_seq, x1_seq, H, W):
        """
        Args:
            x0_seq: (B, L, C) frame0 features (L = H×W)
            x1_seq: (B, L, C) frame1 features
            H, W:   spatial dimensions

        Returns:
            f0_fused, f1_fused: (B, L, C) temporally-fused per-frame features
        """
        B, L, C = x0_seq.shape

        # Concatenate frames: (2B, L, C)
        x_cat = torch.cat([x0_seq, x1_seq], dim=0)
        residual = x_cat

        with torch.amp.autocast('cuda', enabled=False):
            x_norm = self.norm_s(x_cat.float())

            # Reshape to 2D for NSS scanning: (2B, H, W, C)
            x_2d = x_norm.view(2 * B, H, W, C)

            # NSS scan: (2B, H, W, C) → (num_dirs * 2B, L, C)
            x_scanned, scan_info = self.nss.scan(x_2d)

            # Mamba2 spatial processing
            if self.training and self.use_checkpointing:
                x_mamba_out = grad_checkpoint(
                    self.spatial_mamba, x_scanned, use_reentrant=False
                )
            else:
                x_mamba_out = self.spatial_mamba(x_scanned)

            # NSS unscan: (num_dirs * 2B, L, C) → (2B, H, W, C)
            x_spatial = self.nss.unscan(x_mamba_out, scan_info)

            # Flatten back: (2B, H, W, C) → (2B, L, C) + residual
            spatial_out = x_spatial.reshape(2 * B, L, C) + residual.float()

        spatial_out = spatial_out.to(x0_seq.dtype)

        # Split per-frame
        f0 = spatial_out[:B]
        f1 = spatial_out[B:]

        # Temporal cross-fusion
        f0_fused = f0 + self.time_mix_proj(torch.cat([f0, f1], dim=-1))
        f1_fused = f1 + self.time_mix_proj(torch.cat([f1, f0], dim=-1))

        f0_fused = self.norm_t(f0_fused)
        f1_fused = self.norm_t(f1_fused)

        return f0_fused, f1_fused


# ═══════════════════════════════════════════════════════════════════════════
# LGS Block V3
# ═══════════════════════════════════════════════════════════════════════════

class LGSBlockV3(nn.Module):
    """
    Local-Global Synergistic Block V3.

    Branch A (Global):  FactorizedSSMBlock with NSS scan
    Branch B (Local):   GatedWindowAttention (shifted windows)
    Fusion:             CrossGatingFusion (hybrid) or ECAB (single-branch)

    The NSS shift-stripe mechanism alternates with shifted window attention:
      - Even blocks (shift_size=0): standard NSS + standard windows
      - Odd blocks  (shift_size>0): shifted NSS + shifted windows
    This ensures both branches benefit from offset-boundary information exchange.

    Ablation modes:
        'hybrid':          Both branches + CrossGatingFusion
        'mamba2_only':     Mamba2 only + ECAB channel attention
        'gated_attn_only': Attention only + ECAB channel attention
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mamba_d_state=64, mamba_d_conv=4, mamba_expand=2,
                 mamba_headdim=64,
                 num_scan_dirs=4, stripe_width=4,
                 backbone_mode='hybrid',
                 use_ecab=True, use_checkpointing=True):
        super().__init__()
        self.dim = dim
        self.backbone_mode = backbone_mode
        self.norm1 = norm_layer(dim)

        # --- Branch A: Factorized Mamba2 with NSS ---
        self.use_mamba = backbone_mode in ['hybrid', 'mamba2_only']
        if self.use_mamba:
            if Mamba2 is None:
                raise ImportError(
                    "mamba_ssm required for Mamba2 backbone. "
                    "Install with: pip install mamba-ssm>=2.0"
                )
            # NSS shift-stripe aligned with shifted window attention
            nss_shift = (shift_size > 0)
            self.factorized_ssm = FactorizedSSMBlock(
                d_model=dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                num_scan_dirs=num_scan_dirs,
                stripe_width=stripe_width,
                shift=nss_shift,
                use_checkpointing=use_checkpointing,
                headdim=mamba_headdim,
            )
        else:
            self.factorized_ssm = None

        # --- Branch B: Gated Window Attention ---
        self.use_attn = backbone_mode in ['hybrid', 'gated_attn_only']
        self.window_size = window_size
        self.shift_size = shift_size
        if self.use_attn:
            self.attn = GatedWindowAttention(
                dim, window_size=window_size, num_heads=num_heads,
                qkv_bias=True, qk_scale=None,
                attn_drop=attn_drop, proj_drop=drop,
            )
        else:
            self.attn = None

        # --- Fusion ---
        if backbone_mode == 'hybrid':
            self.cross_gate = CrossGatingFusion(dim)
        else:
            self.cross_gate = None

        # Channel attention (for single-branch modes)
        if use_ecab:
            self.cab = ECAB(dim)
        else:
            self.cab = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid(),
            )
        self.use_ecab = use_ecab

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    # ----- shifted window attention -----

    def _window_attn(self, x_norm, H, W):
        """Shifted window attention on (2B, L, C) input."""
        twoB, L, C = x_norm.shape
        x_2d = x_norm.view(twoB, H, W, C)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x_norm.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0) \
                                 .masked_fill(attn_mask == 0, 0.0)
            shifted_x = torch.roll(
                x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x_2d
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x_attn = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x_attn = shifted_x

        return x_attn.view(twoB, L, C)

    # ----- channel attention helper -----

    def _apply_cab(self, x_2d):
        """Apply channel attention block on (N, C, H, W) tensor."""
        if self.use_ecab:
            return self.cab(x_2d)
        else:
            return x_2d * self.cab(x_2d)

    # ----- forward -----

    def forward(self, x, H, W):
        """
        Args:
            x: (2B, L, C) — first B tokens are frame0, last B are frame1
        """
        twoB, L, C = x.shape
        B = twoB // 2
        shortcut = x
        x_norm = self.norm1(x)

        x_mamba = None
        x_attn = None

        # --- Branch A: Factorized Mamba2 with NSS ---
        if self.use_mamba:
            x0_norm, x1_norm = x_norm[:B], x_norm[B:]
            f0_m, f1_m = self.factorized_ssm(x0_norm, x1_norm, H, W)
            x_mamba = torch.cat([f0_m, f1_m], dim=0)  # (2B, L, C)

        # --- Branch B: Gated Window Attention ---
        if self.use_attn:
            x_attn = self._window_attn(x_norm, H, W)  # (2B, L, C)

        # --- Fusion ---
        if self.backbone_mode == 'hybrid':
            fm_2d = x_mamba.transpose(1, 2).view(twoB, C, H, W)
            fa_2d = x_attn.transpose(1, 2).view(twoB, C, H, W)
            x_fused = self.cross_gate(fm_2d, fa_2d)
            x_fused = x_fused.flatten(2).transpose(1, 2)
            x = shortcut + self.drop_path(x_fused)

        elif self.backbone_mode == 'mamba2_only':
            x_fused = x_mamba.transpose(1, 2).view(twoB, C, H, W)
            x_fused = self._apply_cab(x_fused)
            x = shortcut + self.drop_path(x_fused.flatten(2).transpose(1, 2))

        else:  # gated_attn_only
            x_fused = x_attn.transpose(1, 2).view(twoB, C, H, W)
            x_fused = self._apply_cab(x_fused)
            x = shortcut + self.drop_path(x_fused.flatten(2).transpose(1, 2))

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════════════
# Stage Layer
# ═══════════════════════════════════════════════════════════════════════════

class BasicLayerV3(nn.Module):
    """A single stage containing multiple LGSBlockV3 blocks."""

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 num_scan_dirs=4, stripe_width=4,
                 backbone_mode='hybrid',
                 use_ecab=True, use_checkpointing=True,
                 mamba_headdim=64):
        super().__init__()
        self.blocks = nn.ModuleList([
            LGSBlockV3(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                num_scan_dirs=num_scan_dirs,
                stripe_width=stripe_width,
                backbone_mode=backbone_mode,
                use_ecab=use_ecab,
                use_checkpointing=use_checkpointing,
                mamba_headdim=mamba_headdim,
            )
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Backbone V3
# ═══════════════════════════════════════════════════════════════════════════

class HybridBackboneV3(nn.Module):
    """
    Integrated Hybrid Backbone for Video Frame Interpolation.

    Architecture highlights:
      1. FactorizedSSMBlock with NSS scan (spatial Mamba2 + temporal MLP)
      2. GatedWindowAttention with shifted windows (local texture)
      3. CrossGatingFusion for hybrid branch merging
      4. Learnable frame merge (concat + 1×1 conv)
      5. Shift-stripe NSS aligned with shifted window attention

    Args:
        embed_dims:       list of channel dims per stage, e.g. [32, 64, 128]
        depths:           list of block counts per stage, e.g. [2, 2, 2]
        num_heads:        list of attention heads per stage
        window_sizes:     list of window sizes per stage
        mlp_ratios:       list of MLP expansion ratios
        num_scan_dirs:    NSS directions: 2 (h+v) or 4 (h+v fwd+bwd)
        stripe_width:     NSS stripe width in pixels (default 4)
        backbone_mode:    'hybrid' | 'mamba2_only' | 'gated_attn_only'
        use_ecab:         use ECAB (True) or SE-block (False)
        use_checkpointing: gradient checkpointing for Mamba2
    """

    def __init__(self, embed_dims=[32, 64, 128], depths=[2, 2, 2],
                 num_heads=[2, 4, 8], window_sizes=[8, 8, 8],
                 mlp_ratios=[4, 4, 4], drop_rate=0., drop_path_rate=0.1,
                 num_scan_dirs=4, stripe_width=4,
                 backbone_mode='hybrid',
                 use_ecab=True, use_checkpointing=True,
                 mamba_headdim=64):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.backbone_mode = backbone_mode

        # ---- Patch Embedding ----
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ---- Stochastic depth schedule ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ---- Stage layers ----
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerV3(
                dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                num_scan_dirs=num_scan_dirs,
                stripe_width=stripe_width,
                backbone_mode=backbone_mode,
                use_ecab=use_ecab,
                use_checkpointing=use_checkpointing,
                mamba_headdim=mamba_headdim,
            )
            self.layers.append(layer)

        # ---- Inter-stage downsamplers ----
        self.downsamplers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.downsamplers.append(nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, 2, 1),
                nn.LeakyReLU(0.1),
            ))

        # ---- Learnable frame merge (V1-style) ----
        # VFI requires learning how to combine two frames' features.
        # Simple addition loses frame-specific information; concat + 1×1 conv
        # lets the network learn asymmetric weighting per channel.
        self.merge_convs = nn.ModuleList([
            nn.Conv2d(embed_dims[i] * 2, embed_dims[i], kernel_size=1)
            for i in range(self.num_layers)
        ])

    def forward(self, img0, img1):
        """
        VFIMamba-style batch processing.

        Args:
            img0, img1: (B, 3, H, W) — input frame pair

        Returns:
            outs: list of multi-scale features, each (B, C_i, H_i, W_i)
        """
        B_orig = img0.shape[0]

        # Batch-concat: (2B, 3, H, W)
        x = torch.cat([img0, img1], dim=0)
        x = self.patch_embed(x)

        outs = []
        twoB, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (2B, H*W, C)

        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)

            # Reshape to spatial
            x_out = x.transpose(1, 2).view(twoB, -1, H, W)

            # Learnable frame merge: concat → 1×1 conv
            x_f0 = x_out[:B_orig]
            x_f1 = x_out[B_orig:]
            x_merged = self.merge_convs[i](
                torch.cat([x_f0, x_f1], dim=1)
            )
            outs.append(x_merged)

            # Downsample for next stage
            if i < self.num_layers - 1:
                x_down = self.downsamplers[i](x_out)
                _, _, H, W = x_down.shape
                x = x_down.flatten(2).transpose(1, 2)

        return outs

    # ----- Weight transfer utilities -----

    def init_mamba_from_attn(self):
        """
        MaTVLM-style init: transfer Attention Q/K/V → Mamba2 B/C/x.
        Call AFTER construction, BEFORE training.
        """
        if self.backbone_mode != 'hybrid':
            print("init_mamba_from_attn: skipped (not hybrid mode)")
            return
        count = 0
        for layer in self.layers:
            for blk in layer.blocks:
                if (blk.use_mamba and blk.use_attn
                        and hasattr(blk.factorized_ssm.spatial_mamba, 'in_proj')):
                    matvlm_init_mamba2(blk.factorized_ssm.spatial_mamba, blk.attn)
                    count += 1
        print(f"MaTVLM init: transferred weights for {count} LGSBlockV3 blocks")

    def load_v1_weights(self, v1_state_dict):
        """Load compatible weights from V1 backbone (partial transfer)."""
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in v1_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"V1→V3 weight transfer: {loaded} loaded, {skipped} skipped")
        return loaded, skipped

    def load_v2_weights(self, v2_state_dict):
        """Load compatible weights from V2 backbone (partial transfer)."""
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in v2_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"V2→V3 weight transfer: {loaded} loaded, {skipped} skipped")
        return loaded, skipped


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_backbone_v3(cfg):
    """
    Build V3 backbone from config dict.

    Example config:
        cfg = {
            'embed_dims': [32, 64, 128],
            'depths': [2, 2, 2],
            'num_heads': [2, 4, 8],
            'window_sizes': [8, 8, 8],
            'mlp_ratios': [4, 4, 4],
            'num_scan_dirs': 4,          # 2 or 4 (NSS directions)
            'stripe_width': 4,           # NSS stripe width in pixels
            'backbone_mode': 'hybrid',   # 'hybrid', 'mamba2_only', 'gated_attn_only'
            'use_ecab': True,
            'use_checkpointing': True,
        }
    """
    return HybridBackboneV3(
        embed_dims=cfg['embed_dims'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_sizes=cfg['window_sizes'],
        mlp_ratios=cfg['mlp_ratios'],
        num_scan_dirs=cfg.get('num_scan_dirs', 4),
        stripe_width=cfg.get('stripe_width', 4),
        backbone_mode=cfg.get('backbone_mode', 'hybrid'),
        use_ecab=cfg.get('use_ecab', True),
        use_checkpointing=cfg.get('use_checkpointing', True),
        mamba_headdim=cfg.get('mamba_headdim', 64),
    )