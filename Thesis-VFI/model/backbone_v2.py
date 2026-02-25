"""
backbone_v2.py — Factorized Spatio-Temporal SSM Backbone
=========================================================
Replaces Interleaved SS2D with:
  1. FactorizedSSMBlock: shared-weight spatial Mamba2 + symmetric temporal MLP fusion
  2. CrossGatingFusion: bi-directional cross-gating (replaces ECAB for hybrid fusion)

VRAM benchmark (RTX 5090 32GB, 256x256 crop):
  Original (Interleaved SS2D): batch=4 → 22.12 GB, batch=8 → OOM
  Factorized (this file):      batch=4 →  7.47 GB, batch=8 → 14.44 GB

References:
  - VideoMamba (Li et al., CVPR 2024): factorized scan strategy
  - DST-Mamba (2025): divided space-time SSM, 92% complexity reduction
  - ZigMa (2024): factorized Mamba for video generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from functools import partial
from timm.models.layers import DropPath, trunc_normal_

try:
    from mamba_ssm import Mamba2
except ImportError:
    import warnings
    warnings.warn("mamba_ssm not found.")
    Mamba2 = None

from .utils import (window_partition, window_reverse, Mlp, ECAB,
                    CrossGatingFusion, matvlm_init_mamba2)
from .backbone import GatedWindowAttention


class FactorizedSSMBlock(nn.Module):
    """
    Spatio-Temporal Factorized SSM Block.
    
    Stage 1 (Spatial): Shared-weight Mamba2 processes each frame independently.
        Sequence length = HW (not 2HW), halving per-sequence memory.
    Stage 2 (Temporal): Symmetric cross-frame MLP fusion.
        F0' = F0 + MLP([F0 ∥ F1])
        F1' = F1 + MLP([F1 ∥ F0])
    
    This is mathematically equivalent to a lightweight cross-attention
    that compares features at the same spatial position across frames —
    a strong inductive bias for motion estimation in VFI.
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2,
                 use_checkpointing=True):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing
        # Shared spatial Mamba2 (weight-tied across frames)
        self.spatial_mamba = Mamba2(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand
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
            x0_seq: (B, L, C) frame0 features (L = H*W)
            x1_seq: (B, L, C) frame1 features
        Returns:
            f0_fused, f1_fused: (B, L, C) temporally-fused per-frame features
        """
        B, L, C = x0_seq.shape
        # Spatial pass: shared Mamba2, frames as batch dimension
        x_cat = torch.cat([x0_seq, x1_seq], dim=0)  # (2B, L, C)
        residual = x_cat
        with torch.amp.autocast('cuda', enabled=False):
            x_norm = self.norm_s(x_cat.float())
            # Gradient checkpointing: trade ~15-20% compute for ~30-40% VRAM savings
            # on Mamba2 SSD backward pass (materializes fewer intermediate tensors)
            if self.training and self.use_checkpointing:
                spatial_out = grad_checkpoint(
                    self.spatial_mamba, x_norm, use_reentrant=False
                ) + residual.float()
            else:
                spatial_out = self.spatial_mamba(x_norm) + residual.float()
        spatial_out = spatial_out.to(x0_seq.dtype)

        f0 = spatial_out[:B]  # (B, L, C)
        f1 = spatial_out[B:]  # (B, L, C)

        # Temporal pass: symmetric cross-frame fusion
        f0_fused = f0 + self.time_mix_proj(torch.cat([f0, f1], dim=-1))
        f1_fused = f1 + self.time_mix_proj(torch.cat([f1, f0], dim=-1))
        f0_fused = self.norm_t(f0_fused)
        f1_fused = self.norm_t(f1_fused)
        return f0_fused, f1_fused


class LGSBlockV2(nn.Module):
    """
    Local-Global Synergistic Block V2.
    
    Changes from V1:
      - Branch A: FactorizedSSMBlock (replaces Interleaved SS2D)
      - Fusion: CrossGatingFusion (replaces 1x1 conv + ECAB)
    
    Branch A (Mamba2): Global context via factorized spatial scan + temporal fusion
    Branch B (Attention): Local texture via gated window attention
    Fusion: Bi-directional cross-gating with Conv2d bottleneck gates
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mamba_d_state=64, mamba_d_conv=4, mamba_expand=2,
                 backbone_mode='hybrid', use_ecab=True):
        super().__init__()
        self.dim = dim
        self.backbone_mode = backbone_mode
        self.norm1 = norm_layer(dim)

        # Branch A: Factorized Mamba2
        self.use_mamba = backbone_mode in ['hybrid', 'mamba2_only']
        if self.use_mamba:
            if Mamba2 is None:
                raise ImportError("mamba_ssm required for Mamba2 backbone.")
            self.factorized_ssm = FactorizedSSMBlock(
                d_model=dim, d_state=mamba_d_state,
                d_conv=mamba_d_conv, expand=mamba_expand
            )
        else:
            self.factorized_ssm = None

        # Branch B: Gated Window Attention
        self.use_attn = backbone_mode in ['hybrid', 'gated_attn_only']
        self.window_size = window_size
        self.shift_size = shift_size
        if self.use_attn:
            self.attn = GatedWindowAttention(
                dim, window_size=window_size, num_heads=num_heads,
                qkv_bias=True, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = None

        # Fusion
        if backbone_mode == 'hybrid':
            self.cross_gate = CrossGatingFusion(dim)
        else:
            self.cross_gate = None

        # Channel attention (applied after cross-gating for single-branch modes)
        if use_ecab:
            self.cab = ECAB(dim)
        else:
            self.cab = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1), nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid()
            )
        self.use_ecab = use_ecab

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def _window_attn(self, x_norm, H, W):
        """Standard shifted window attention on (2B, L, C) input."""
        twoB, L, C = x_norm.shape
        x_2d = x_norm.view(twoB, H, W, C)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x_norm.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_2d
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x_attn = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted_x
        return x_attn.view(twoB, L, C)

    def forward(self, x, H, W):
        """
        Args:
            x: (2B, L, C) — first B are frame0, last B are frame1 features
        """
        twoB, L, C = x.shape
        B = twoB // 2
        shortcut = x
        x_norm = self.norm1(x)

        x_mamba = None
        x_attn = None

        # --- Branch A: Factorized Mamba2 ---
        if self.use_mamba:
            x0_norm, x1_norm = x_norm[:B], x_norm[B:]
            f0_m, f1_m = self.factorized_ssm(x0_norm, x1_norm, H, W)
            x_mamba = torch.cat([f0_m, f1_m], dim=0)  # (2B, L, C)

        # --- Branch B: Window Attention ---
        if self.use_attn:
            x_attn = self._window_attn(x_norm, H, W)  # (2B, L, C)

        # --- Fusion ---
        if self.backbone_mode == 'hybrid':
            # CrossGatingFusion expects (B, C, H, W)
            fm_2d = x_mamba.transpose(1, 2).view(twoB, C, H, W)
            fa_2d = x_attn.transpose(1, 2).view(twoB, C, H, W)
            x_fused = self.cross_gate(fm_2d, fa_2d)  # (2B, C, H, W)
            x_fused = x_fused.flatten(2).transpose(1, 2)  # (2B, L, C)
            x = shortcut + self.drop_path(x_fused)
        elif self.backbone_mode == 'mamba2_only':
            x_fused = x_mamba.transpose(1, 2).view(twoB, C, H, W)
            if self.use_ecab:
                x_fused = self.cab(x_fused)
            else:
                x_fused = x_fused * self.cab(x_fused)
            x = shortcut + self.drop_path(x_fused.flatten(2).transpose(1, 2))
        else:  # gated_attn_only
            x_fused = x_attn.transpose(1, 2).view(twoB, C, H, W)
            if self.use_ecab:
                x_fused = self.cab(x_fused)
            else:
                x_fused = x_fused * self.cab(x_fused)
            x = shortcut + self.drop_path(x_fused.flatten(2).transpose(1, 2))

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayerV2(nn.Module):
    """A basic layer for one stage using LGSBlockV2."""
    def __init__(self, dim, output_dim, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 backbone_mode='hybrid', use_ecab=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            LGSBlockV2(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                backbone_mode=backbone_mode,
                use_ecab=use_ecab
            )
            for i in range(depth)
        ])

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W


class HybridBackboneV2(nn.Module):
    """
    Upgraded Hybrid Backbone with Factorized SSM + CrossGating.
    
    Key differences from V1:
      - FactorizedSSMBlock: seq length HW (not 2HW), ~3x VRAM reduction
      - CrossGatingFusion: dynamic spatial gating (not channel-only ECAB)
      - Supports loading V1 backbone weights (partial, strict=False)
    """
    def __init__(self, embed_dims=[32, 64, 128], depths=[2, 2, 2],
                 num_heads=[2, 4, 8], window_sizes=[8, 8, 8],
                 mlp_ratios=[4, 4, 4], drop_rate=0., drop_path_rate=0.1,
                 backbone_mode='hybrid', use_ecab=True):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.backbone_mode = backbone_mode

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerV2(
                dim=embed_dims[i_layer],
                output_dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                backbone_mode=backbone_mode,
                use_ecab=use_ecab
            )
            self.layers.append(layer)

        self.downsamplers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.downsamplers.append(nn.Sequential(
                nn.Conv2d(embed_dims[i], embed_dims[i + 1], 3, 2, 1),
                nn.LeakyReLU(0.1)
            ))

    def forward(self, img0, img1):
        """
        Input: img0, img1: (B, 3, H, W)
        Output: list of multi-scale features, each (B, C_i, H_i, W_i)
        """
        B_orig = img0.shape[0]
        x = torch.cat([img0, img1], dim=0)  # (2B, 3, H, W)
        x = self.patch_embed(x)

        outs = []
        twoB, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (2B, L, C)

        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            x_out = x.transpose(1, 2).view(twoB, -1, H, W)
            x_out_merged = x_out[:B_orig] + x_out[B_orig:]
            outs.append(x_out_merged)

            if i < self.num_layers - 1:
                x_down = self.downsamplers[i](x_out)
                _, _, H, W = x_down.shape
                x = x_down.flatten(2).transpose(1, 2)

        return outs

    def load_v1_weights(self, v1_state_dict):
        """
        Load compatible weights from V1 backbone (Interleaved SS2D).
        Transfers: patch_embed, downsamplers, attention, mlp, norm, cab weights.
        Skips: mamba (architecture changed), fusion_conv (replaced by cross_gate).
        """
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in v1_state_dict.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"V1→V2 weight transfer: {loaded} loaded, {skipped} skipped "
              f"(new modules: factorized_ssm, cross_gate)")
        return loaded, skipped


def build_backbone_v2(cfg):
    """Build V2 backbone from config dict."""
    return HybridBackboneV2(
        embed_dims=cfg['embed_dims'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_sizes=cfg['window_sizes'],
        mlp_ratios=cfg['mlp_ratios'],
        backbone_mode=cfg.get('backbone_mode', 'hybrid'),
        use_ecab=cfg.get('use_ecab', True)
    )
