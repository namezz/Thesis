import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple, trunc_normal_

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

def scan_images(x):
    """
    SS2D Scan: 4 directions (Original, Flip H, Flip V, Flip HV)
    Input: (B, H, W, C)
    Output: (4*B, L, C) where L = H*W
    """
    B, H, W, C = x.shape
    
    # 1. Original
    x0 = x.flatten(1, 2) # (B, L, C)
    
    # 2. Flip H
    x1 = x.flip(2).flatten(1, 2)
    
    # 3. Flip V
    x2 = x.flip(1).flatten(1, 2)
    
    # 4. Flip HV (Rotate 180)
    x3 = x.flip(1, 2).flatten(1, 2)
    
    # Batch them together for Mamba efficiency
    x_scan = torch.cat([x0, x1, x2, x3], dim=0) # (4B, L, C)
    return x_scan

def merge_images(x_scan, B, H, W):
    """
    SS2D Merge: Reverse scanning and sum/average
    Input: (4*B, L, C)
    Output: (B, H, W, C)
    """
    C = x_scan.shape[-1]
    
    # Split back to 4 branches
    x0, x1, x2, x3 = x_scan.chunk(4, dim=0) # Each (B, L, C)
    
    # Reshape and un-flip
    x0 = x0.view(B, H, W, C)
    x1 = x1.view(B, H, W, C).flip(2)
    x2 = x2.view(B, H, W, C).flip(1)
    x3 = x3.view(B, H, W, C).flip(1, 2)
    
    return x0 + x1 + x2 + x3


def interleaved_scan(x):
    """
    VFIMamba-style Interleaved SS2D Scan for cross-frame temporal interaction.
    Ref: VFIMamba (NeurIPS 2024, Zhang et al.)

    Input: x of shape (2B, H, W, C) — first B samples are frame0 features,
           last B samples are frame1 features (batch-concatenated).
    Output: (4B, L, C) where L = 2*H*W (interleaved tokens from both frames)

    Key insight: Interleaving tokens from two frames into one sequence lets the
    SSM's recurrent state carry information across frames, enabling cross-frame
    temporal modeling within a single Mamba pass.

    4 scan directions:
      1. H→W order (interleaved)
      2. W→H order (transposed, interleaved)
      3. Reverse of direction 1
      4. Reverse of direction 2
    """
    twoB, H, W, C = x.shape
    B = twoB // 2
    L = 2 * H * W
    
    # Convert to (2B, C, H, W) for spatial operations
    x_perm = x.permute(0, 3, 1, 2).contiguous()  # (2B, C, H, W)

    def _merge_frames(feat):
        """Interleave frame0 and frame1 tokens: (2B, C, H, W) -> (B, C, 2*H*W)"""
        # feat: (2B, C, H, W)
        feat_flat = feat.reshape(twoB, C, H * W).transpose(1, 2)  # (2B, H*W, C)
        # Interleave: concat frame0[i] and frame1[i] along sequence dim
        merged = torch.cat([feat_flat[:B], feat_flat[B:]], dim=-1)  # (B, H*W, 2C)
        merged = merged.reshape(B, L, C)  # (B, 2*H*W, C)
        return merged.transpose(1, 2).contiguous()  # (B, C, L)

    # Direction 1: H→W scan
    d1 = _merge_frames(x_perm)  # (B, C, L)
    # Direction 2: W→H scan (transpose spatial dims)
    x_trans = x_perm.transpose(2, 3).contiguous()  # (2B, C, W, H)
    d2 = _merge_frames(x_trans)  # (B, C, L)

    # Stack directions 1 & 2, then add their reverses as directions 3 & 4
    x_hwwh = torch.stack([d1, d2], dim=1)  # (B, 2, C, L)
    xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, C, L)

    # Reshape to (4B, L, C) for Mamba processing
    xs = xs.reshape(4 * B, C, L).transpose(1, 2).contiguous()  # (4B, L, C)
    return xs


def interleaved_merge(x_scan, B, H, W):
    """
    VFIMamba-style Interleaved SS2D Merge: reverse the interleaved scanning.
    Input: (4B, L, C) where L = 2*H*W
    Output: (2B, H, W, C) — first B are frame0, last B are frame1

    Reverses the 4 scan directions and de-interleaves the merged sequences
    back to per-frame feature maps.
    """
    L = 2 * H * W
    C = x_scan.shape[-1]

    # (4B, L, C) -> (B, 4, C, L)
    out = x_scan.transpose(1, 2).contiguous().reshape(B, 4, C, L)
    
    # Reverse directions 3 & 4
    out[:, 2] = torch.flip(out[:, 2], dims=[-1])
    out[:, 3] = torch.flip(out[:, 3], dims=[-1])

    def _unmerge_frames(merged, h, w):
        """De-interleave: (B, C, 2*h*w) -> (2B, C, h, w)"""
        # merged: (B, C, L) where L = 2*h*w
        merged_seq = merged.transpose(1, 2).contiguous()  # (B, L, C)
        merged_seq = merged_seq.reshape(B, h * w, 2, C)  # (B, h*w, 2, C)
        f0 = merged_seq[:, :, 0, :]  # (B, h*w, C)
        f1 = merged_seq[:, :, 1, :]  # (B, h*w, C)
        f0 = f0.transpose(1, 2).reshape(B, C, h, w)
        f1 = f1.transpose(1, 2).reshape(B, C, h, w)
        return torch.cat([f0, f1], dim=0)  # (2B, C, h, w)

    # Direction 1 (H→W) + Direction 3 (reverse of 1): sum and de-interleave
    y_hw = _unmerge_frames(out[:, 0] + out[:, 2], H, W)  # (2B, C, H, W)
    
    # Direction 2 (W→H) + Direction 4 (reverse of 2): sum, transpose back, de-interleave
    y_wh_merged = out[:, 1] + out[:, 3]  # (B, C, L)
    y_wh = _unmerge_frames(y_wh_merged, W, H)  # (2B, C, W, H) — note: W, H swapped
    y_wh = y_wh.transpose(2, 3).contiguous()  # (2B, C, H, W) — transpose back

    # Sum all directions
    y = y_hw + y_wh  # (2B, C, H, W)
    return y.permute(0, 2, 3, 1).contiguous()  # (2B, H, W, C)


def sinkhorn_knopp(W, iterations=20):
    """
    Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff Polytope 
    (doubly stochastic matrices).
    Input W: (..., M, N)
    """
    for _ in range(iterations):
        W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)
        W = W / (W.sum(dim=-2, keepdim=True) + 1e-6)
    return W


class ManifoldResConnection(nn.Module):
    """
    mHC: Manifold-Constrained Hyper-Connections (DeepSeek, Xie et al., 2025)
    Constrains residual mixing matrix to the Birkhoff Polytope using Sinkhorn-Knopp.
    """
    def __init__(self, dim, num_streams=2):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.mixing_weights = nn.Parameter(torch.eye(num_streams))
        
    def forward(self, streams):
        """
        streams: List of [B, L, C] tensors
        """
        W = torch.exp(self.mixing_weights) 
        W_stochastic = sinkhorn_knopp(W)
        
        out = 0
        for j in range(self.num_streams):
            out = out + W_stochastic[0, j] * streams[j]
            
        return out


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