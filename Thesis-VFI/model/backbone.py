import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm import Mamba2
except ImportError:
    print("Warning: Mamba2 not found. Using placeholder or falling back.")
    Mamba2 = None # Handle gracefully or ensure env is correct

from .utils import window_partition, window_reverse, Mlp

class GatedWindowAttention(nn.Module):
    """
    Window based multi-head self-attention (W-MSA) with gating mechanism.
    Ref: Gated Attention (NeurIPS 2025)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Gating parameter: one per head
        self.gate = nn.Linear(dim, num_heads) 
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Standard Attention Output
        x_attn = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Gated Attention Mechanism
        # Gate calculation: (B_, N, C) -> (B_, N, num_heads)
        gate_score = self.sigmoid(self.gate(x)) 
        # Expand gate to match C: (B, N, heads, 1) * (B, N, heads, head_dim) logic roughly
        # Reshape gate to apply per head
        gate_score = gate_score.unsqueeze(-1) # (B, N, heads, 1)
        
        # We need to apply this to the head output before reshaping back to C, 
        # or we can apply it after if we reshape x_attn back to heads.
        # Let's do it before the final reshape in a real implementation, but here x_attn is already (B,N,C).
        # Let's reshape x_attn to (B, N, heads, head_dim) to apply gate
        x_attn_heads = x_attn.view(B_, N, self.num_heads, C // self.num_heads)
        x_attn_gated = x_attn_heads * gate_score
        x_attn = x_attn_gated.reshape(B_, N, C)

        x = self.proj(x_attn)
        x = self.proj_drop(x)
        return x

class LGSBlock(nn.Module):
    """
    Local-Global Synergistic Block
    Branch A: Mamba2 (SSD) for Global Context
    Branch B: Gated Window Attention for Local Texture
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mamba_d_state=64, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        
        # Branch A: Mamba2
        # Note: Mamba2 expects (B, L, C). We will flatten spatial dims.
        if Mamba2 is not None:
            self.mamba = Mamba2(
                d_model=dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand
            )
        else:
            self.mamba = nn.Identity() # Fallback

        # Branch B: Gated Window Attention
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = GatedWindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=True, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)

        # Fusion
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1) # 1x1 conv to merge branches
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # x: (B, L, C) where L = H*W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        # --- Branch A: Mamba2 (Global) ---
        # Mamba2 handles (B, L, C) directly.
        # TODO: Add 2D-scanning (interleaved or zig-zag) here for better spatial awareness if needed.
        # For Phase 1 baseline, we treat it as a sequence.
        x_mamba = self.mamba(x)
        
        # --- Branch B: Window Attention (Local) ---
        x_2d = x.view(B, H, W, C)
        
        # Shift Window
        if self.shift_size > 0:
            shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_2d

        # Partition
        x_windows = window_partition(shifted_x, self.window_size) # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # (nW*B, window_size*window_size, C)
        
        # Attention
        # TODO: Add mask for shifted window if needed (omitted for brevity in baseline)
        attn_windows = self.attn(x_windows) 
        
        # Reverse Partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse Shift
        if self.shift_size > 0:
            x_attn = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted_x
            
        x_attn = x_attn.view(B, L, C)
        
        # --- Fusion ---
        # Concat along channel: (B, L, 2C) -> Transpose to (B, 2C, H, W) -> 1x1 Conv -> (B, C, H, W) -> (B, L, C)
        x_cat = torch.cat([x_mamba, x_attn], dim=-1)
        x_cat = x_cat.transpose(1, 2).view(B, 2*C, H, W)
        x_fused = self.fusion_conv(x_cat).flatten(2).transpose(1, 2)
        
        x = shortcut + self.drop_path(x_fused)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class BasicLayer(nn.Module):
    """ A basic Swin/Mamba layer for one stage """
    def __init__(self, dim, output_dim, depth, num_heads, window_size, mlp_ratio=4., drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            LGSBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Patch Merging / Downsample
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return x, H, W

class HybridBackbone(nn.Module):
    """
    Phase 1: Hybrid Backbone Baseline
    """
    def __init__(self, embed_dims=[32, 64, 128], depths=[2, 2, 2], num_heads=[2, 4, 8], 
                 window_sizes=[7, 7, 7], mlp_ratios=[4, 4, 4], drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patch_embed = nn.Sequential(
            nn.Conv2d(6, embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                output_dim=embed_dims[i_layer], # Keep dim same in block, handle scale changes if needed
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=nn.LayerNorm,
                downsample=None # For Phase 1, we keep resolution or use simple strided convs between stages if needed.
                # NOTE: For VFI, usually we want multi-scale features.
                # Let's assume we maintain resolution or have explicit downsampling steps.
            )
            self.layers.append(layer)
            
        # Simple downsamplers for multi-scale hierarchy
        self.downsamplers = nn.ModuleList()
        for i in range(self.num_layers - 1):
             self.downsamplers.append(nn.Sequential(
                 nn.Conv2d(embed_dims[i], embed_dims[i+1], 3, 2, 1),
                 nn.LeakyReLU(0.1)
             ))

    def forward(self, img0, img1):
        x = torch.cat((img0, img1), 1)
        x = self.patch_embed(x) # (B, C, H, W)
        
        outs = []
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, L, C)
        
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            # Output features of this stage
            x_out = x.transpose(1, 2).view(B, -1, H, W)
            outs.append(x_out)
            
            if i < self.num_layers - 1:
                # Downsample for next stage
                x_down = self.downsamplers[i](x_out)
                _, _, H, W = x_down.shape
                x = x_down.flatten(2).transpose(1, 2)
                B = x.shape[0]

        return outs # List of [Scale1, Scale2, Scale3] features

def build_backbone(cfg):
    # Map config to arguments
    return HybridBackbone(
        embed_dims=cfg['embed_dims'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_sizes=cfg['window_sizes'],
        mlp_ratios=cfg['mlp_ratios']
    )