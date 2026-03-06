"""
utils.py — Shared modules for VFI backbone stages
====================================================
Active modules (used by backbone_v1/v2/v3):
  - window_partition / window_reverse   — GatedWindowAttention
  - Mlp                                 — Attention block FFN
  - ECAB                                — channel attention (ECA-Net)
  - CrossGatingFusion                   — Mamba2 + Attention branch fusion
  - matvlm_init_mamba2                  — weight transfer init

V1-only modules (kept for backward compat):
  - scan_images / merge_images          — 4-dir snake SS2D scan
  - ManifoldResConnection               — mHC ablation candidate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple, trunc_normal_


# ════════════════════════════════════════════════════════════════════════════
# Window Utilities (all backbones)
# ════════════════════════════════════════════════════════════════════════════

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    """Feed-forward network for attention blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ════════════════════════════════════════════════════════════════════════════
# Channel Attention (all backbones)
# ════════════════════════════════════════════════════════════════════════════

class ECAB(nn.Module):
    """
    Efficient Channel Attention Block (ECA-Net, CVPR 2020)
    Replaces standard CAB (SE-Block) for better efficiency and performance.
    Avoids dimensionality reduction and uses 1D convolution for cross-channel interaction.
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECAB, self).__init__()
        # Adaptive kernel size k
        k = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        self.padding = kernel_size // 2
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W) or (B, L, C) -> needs handling
        is_sequence = False
        if x.dim() == 3: # (B, L, C)
            is_sequence = True
            B, L, C = x.shape
            # Safety: L should be window_size² or H*W for a single frame.
            # If L = 2*H*W (interleaved two-frame sequence), this pooling
            # would incorrectly mix frames. Caller must ensure single-frame input.
            y = x.mean(1).view(B, 1, C) # (B, 1, C)
        else:
            y = self.avg_pool(x) # (B, C, 1, 1)
            y = y.squeeze(-1).permute(0, 2, 1) # (B, 1, C)

        y = self.conv(y) # (B, 1, C)
        y = self.sigmoid(y) # (B, 1, C)
        y = y.permute(0, 2, 1) # (B, C, 1)
        
        if is_sequence:
            return x * y.transpose(1, 2) # (B, L, C) * (B, 1, C)
        else:
            return x * y.unsqueeze(-1) # (B, C, H, W) * (B, C, 1, 1)


# ════════════════════════════════════════════════════════════════════════════
# Snake SS2D Scan (backbone V1 only — V3 uses NSScan internally)
# ════════════════════════════════════════════════════════════════════════════

def _snake_flatten(x):
    """
    Snake (S-shaped) scan: alternate L→R and R→L per row.
    Maintains spatial locality by avoiding row-end-to-row-start jumps.
    Ref: LC-Mamba (CVPR 2025), MaIR (2025)
    
    Input: (B, H, W, C)
    Output: (B, H*W, C)
    """
    B, H, W, C = x.shape
    # Flip odd rows horizontally → snake pattern
    x_snake = x.clone()
    x_snake[:, 1::2, :, :] = x_snake[:, 1::2, :, :].flip(2)
    return x_snake.flatten(1, 2)  # (B, L, C)


def _snake_unflatten(x_flat, H, W):
    """
    Reverse snake scan: restore original pixel order.
    Input: (B, H*W, C)
    Output: (B, H, W, C)
    """
    B, L, C = x_flat.shape
    x_2d = x_flat.view(B, H, W, C).clone()
    x_2d[:, 1::2, :, :] = x_2d[:, 1::2, :, :].flip(2)
    return x_2d


def scan_images(x):
    """
    SS2D Snake Scan: 4 directions with S-shaped traversal.
    Each direction uses snake (zigzag) pattern for better spatial locality.
    Ref: LC-Mamba (CVPR 2025), MaIR (2025) — reduces pixel discontinuity
    
    Input: (B, H, W, C)
    Output: (4*B, L, C) where L = H*W
    """
    B, H, W, C = x.shape
    
    # 1. Original snake (T→B, alternating L→R / R→L)
    x0 = _snake_flatten(x)                    # (B, L, C)
    
    # 2. Flip H then snake (T→B, alternating R→L / L→R)
    x1 = _snake_flatten(x.flip(2))            # (B, L, C)
    
    # 3. Flip V then snake (B→T, alternating L→R / R→L)
    x2 = _snake_flatten(x.flip(1))            # (B, L, C)
    
    # 4. Flip HV then snake (B→T, alternating R→L / L→R)
    x3 = _snake_flatten(x.flip(1, 2))         # (B, L, C)
    
    # Batch them together for Mamba efficiency
    x_scan = torch.cat([x0, x1, x2, x3], dim=0) # (4B, L, C)
    return x_scan


def merge_images(x_scan, B, H, W):
    """
    SS2D Snake Merge: Reverse snake scanning and sum.
    Input: (4*B, L, C)
    Output: (B, H, W, C)
    """
    C = x_scan.shape[-1]
    
    # Split back to 4 branches
    x0, x1, x2, x3 = x_scan.chunk(4, dim=0) # Each (B, L, C)
    
    # Reverse snake + un-flip
    x0 = _snake_unflatten(x0, H, W)
    x1 = _snake_unflatten(x1, H, W).flip(2)
    x2 = _snake_unflatten(x2, H, W).flip(1)
    x3 = _snake_unflatten(x3, H, W).flip(1, 2)
    
    return x0 + x1 + x2 + x3



# ════════════════════════════════════════════════════════════════════════════
# Manifold Res-Connection (backbone V1 ablation — not used by V3)
# ════════════════════════════════════════════════════════════════════════════

def sinkhorn_log(logits, num_iters=10, tau=0.05):
    """
    Log-space Sinkhorn-Knopp: project logits onto Birkhoff Polytope (doubly stochastic).
    Ref: mHC (Xie et al., 2025), numerically stable log-space implementation.
    
    Args:
        logits: (..., M, N) raw learnable logits
        num_iters: Sinkhorn iterations (default 10)
        tau: temperature (default 0.05, lower = sharper)
    Returns:
        doubly stochastic matrix (rows & cols sum to 1, all >= 0)
    """
    n = logits.shape[-1]
    Z = logits / tau
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)

    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)

    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class ManifoldResConnection(nn.Module):
    """
    mHC: Manifold-Constrained Hyper-Connections (DeepSeek, Xie et al., 2025)
    Ref: hyper_connections_mhc.py from lucidrains/hyper-connections
    
    Proper implementation with 3 learnable matrices:
    - H_res: residual stream mixing (doubly stochastic via Sinkhorn)
    - H_pre: branch input selection (softmax)
    - H_post: branch output routing (softmax)
    
    Equation: x_{l+1} = H_res @ x_l + H_post^T * F(H_pre @ x_l)
    
    For VFI LGSBlock usage (num_streams=3):
        streams = [shortcut, mamba_out, attn_out]
        The mHC learns to optimally mix/route these signals.
    """
    def __init__(self, dim, num_streams=3, layer_index=0,
                 mhc_num_iters=10, mhc_tau=0.05):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.mhc_num_iters = mhc_num_iters
        self.mhc_tau = mhc_tau
        
        # H_res: residual mixing matrix (streams x streams)
        # Init: near-identity (off-diag=-8 → exp(-8/tau)≈0, diag=0 → balanced)
        init_h_res = torch.full((num_streams, num_streams), -8.0)
        init_h_res.fill_diagonal_(0.0)
        self.H_res_logits = nn.Parameter(init_h_res)
        
        # H_pre: branch input selection (1 x streams)
        # Init: select stream at layer_index
        init_h_pre = torch.full((1, num_streams), -8.0)
        init_h_pre[:, layer_index % num_streams] = 0.0
        self.H_pre_logits = nn.Parameter(init_h_pre)
        
        # H_post: route branch output back to streams (1 x streams)
        # Init: uniform distribution (all zeros → equal softmax)
        self.H_post_logits = nn.Parameter(torch.zeros(1, num_streams))
        
    def forward(self, streams, branch_fn=None):
        """
        Args:
            streams: List of num_streams tensors, each [B, L, C]
            branch_fn: Optional callable. If None, returns (branch_input, add_residual_fn).
                       If provided, applies it and returns fused output.
        Returns:
            If branch_fn is None: (branch_input, add_residual_fn)
            If branch_fn provided: fused residual tensor [B, L, C]
        """
        S = self.num_streams
        # Stack streams: (B, L, S, C)
        x = torch.stack(streams, dim=-2)
        
        # === Width Connection ===
        # H_res: mix residual streams (doubly stochastic)
        H_res = sinkhorn_log(self.H_res_logits, self.mhc_num_iters, self.mhc_tau)  # (S, S)
        residuals = torch.einsum('st,...sd->...td', H_res, x)  # (B, L, S, C)
        
        # H_pre: select branch input (softmax → weighted sum of streams)
        H_pre = self.H_pre_logits.softmax(dim=-1)  # (1, S)
        branch_input = torch.einsum('vs,...sd->...vd', H_pre, x)  # (B, L, 1, C)
        branch_input = branch_input.squeeze(-2)  # (B, L, C)
        
        # H_post: prepare routing weights for depth connection
        H_post = self.H_post_logits.softmax(dim=-1)  # (1, S)
        
        # === Depth Connection ===
        def add_residual_fn(branch_out):
            # Route branch output to streams: (B, L, C) → (B, L, S, C)
            routed = torch.einsum('...d,s->...sd', branch_out, H_post[0])
            # Add to mixed residuals
            fused = residuals + routed  # (B, L, S, C)
            # Sum over streams to collapse
            return fused.sum(dim=-2)  # (B, L, C)
        
        if branch_fn is None:
            return branch_input, add_residual_fn
        
        branch_out = branch_fn(branch_input)
        return add_residual_fn(branch_out)


# ════════════════════════════════════════════════════════════════════════════
# Cross-Gating Fusion (all backbones — hybrid Mamba+Attention branch fusion)
# ════════════════════════════════════════════════════════════════════════════

class CrossGatingFusion(nn.Module):
    """
    Bi-directional Cross-Gating Fusion for Mamba2 + Attention branch outputs.
    Upgraded with 3x3 Depthwise Convolutions for spatial-aware gating.
    Optimized to project to output dimension directly.
    """
    def __init__(self, d_model, d_out=None):
        super().__init__()
        self.d_model = d_model
        d_out = d_out or d_model
        
        # Gate: Mamba features -> gate for Attention branch (Spatial-aware)
        self.gate_mamba_to_attn = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=1)
        )
        # Gate: Attention features -> gate for Mamba branch (Spatial-aware)
        self.gate_attn_to_mamba = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=1)
        )
        # Concatenate gated features and project back to FULL dimension
        self.out_proj = nn.Conv2d(d_model * 2, d_out, kernel_size=1)

    def forward(self, f_mamba, f_attn):
        gate_from_mamba = torch.sigmoid(self.gate_mamba_to_attn(f_mamba))
        gate_from_attn = torch.sigmoid(self.gate_attn_to_mamba(f_attn))
        f_mamba_gated = f_mamba * gate_from_attn
        f_attn_gated = f_attn * gate_from_mamba
        return self.out_proj(torch.cat([f_mamba_gated, f_attn_gated], dim=1))


# ════════════════════════════════════════════════════════════════════════════
# MaTVLM Weight Transfer Init (all backbones — Attention→Mamba2 init)
# ════════════════════════════════════════════════════════════════════════════

def matvlm_init_mamba2(mamba_layer, attn_layer):
    """
    MaTVLM-style initialization: transfer Attention Q/K/V weights to Mamba2 B/C/x.
    Ref: MaTVLM (arXiv:2503.13440, Li et al., HUST, 2025)

    Standard mamba_ssm.Mamba2 in_proj layout: [z(d_inner), x(d_inner), B(ngroups*d_state), C(ngroups*d_state), dt(nheads)]
    Attention QKV layout: qkv = Linear(dim, 3*dim) -> Q(dim), K(dim), V(dim)

    Since d_inner = expand * dim (typically 2*dim) and B/C = ngroups*d_state,
    dimensions may not match exactly. We use partial copy for mismatched dims.
    """
    with torch.no_grad():
        dim = attn_layer.dim
        # Extract Q, K, V weights from the fused qkv linear layer
        qkv_weight = attn_layer.qkv.weight.data  # (3*dim, dim)
        q_weight = qkv_weight[:dim, :]       # (dim, dim)
        k_weight = qkv_weight[dim:2*dim, :]  # (dim, dim)
        v_weight = qkv_weight[2*dim:, :]     # (dim, dim)

        mamba = mamba_layer
        d_inner = mamba.d_inner
        ngroups = mamba.ngroups
        d_state = mamba.d_state
        nheads = mamba.nheads

        # in_proj: [z(d_inner), x(d_inner), B(ngroups*d_state), C(ngroups*d_state), dt(nheads)]
        in_w = mamba.in_proj.weight.data  # (d_in_proj, dim)

        # V -> x section: in_proj[d_inner : d_inner + d_inner]
        x_rows = min(d_inner, dim)
        in_w[d_inner:d_inner + x_rows, :dim] = v_weight[:x_rows, :dim]

        # K -> B section: in_proj[2*d_inner : 2*d_inner + ngroups*d_state]
        b_dim = ngroups * d_state
        b_rows = min(b_dim, dim)
        b_cols = min(dim, in_w.shape[1])
        in_w[2*d_inner:2*d_inner + b_rows, :b_cols] = k_weight[:b_rows, :b_cols]

        # Q -> C section: in_proj[2*d_inner + b_dim : 2*d_inner + 2*b_dim]
        c_rows = min(b_dim, dim)
        in_w[2*d_inner + b_dim:2*d_inner + b_dim + c_rows, :b_cols] = q_weight[:c_rows, :b_cols]

        # out_proj <- attn.proj (partial copy if dim mismatch)
        out_rows = min(mamba.out_proj.weight.shape[0], attn_layer.proj.weight.shape[0])
        out_cols = min(mamba.out_proj.weight.shape[1], attn_layer.proj.weight.shape[1])
        mamba.out_proj.weight.data[:out_rows, :out_cols] = attn_layer.proj.weight.data[:out_rows, :out_cols]
        if mamba.out_proj.bias is not None and attn_layer.proj.bias is not None:
            mamba.out_proj.bias.data[:out_rows] = attn_layer.proj.bias.data[:out_rows]