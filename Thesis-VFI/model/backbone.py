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

from .utils import window_partition, window_reverse, Mlp, ECAB, scan_images, merge_images, interleaved_scan, interleaved_merge, ManifoldResConnection, matvlm_init_mamba2

class GatedWindowAttention(nn.Module):
    """
    Window based multi-head self-attention (W-MSA) with gating mechanism.
    Ref: Gated Attention (arXiv:2505.06708, Qiu et al., Qwen Team, 2025)
    Uses FlashAttention-2 via F.scaled_dot_product_attention.
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

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = mask
        if mask is not None:
            # mask shape: (nW, N, N) where nW is number of windows
            # B_ = BatchSize * nW
            nW = mask.shape[0]
            # Expand mask to (BatchSize*nW, 1, N, N) to match q, k, v batch dim
            # We repeat the mask for each image in the batch
            attn_mask = mask.unsqueeze(1).unsqueeze(0) # (1, nW, 1, N, N)
            attn_mask = attn_mask.repeat(B_ // nW, 1, 1, 1, 1) # (B, nW, 1, N, N)
            attn_mask = attn_mask.view(-1, 1, N, N) # (B_, 1, N, N)

        # FlashAttention-2 (Scaled Dot Product Attention)
        # dropout_p is handled internally. mask can be additive or boolean.
        x_attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.attn_drop.p if self.training else 0., 
            scale=self.scale
        )
        
        # Reshape back: (B_, num_heads, N, head_dim) -> (B_, N, num_heads, head_dim) -> (B_, N, C)
        x_attn = x_attn.transpose(1, 2).reshape(B_, N, C)
        
        # Gated Attention Mechanism
        # Gate calculation: (B_, N, C) -> (B_, N, num_heads)
        gate_score = self.sigmoid(self.gate(x)) 
        # Reshape gate to apply per head: (B_, N, heads, 1)
        gate_score = gate_score.unsqueeze(-1) 
        
        # Apply gate
        x_attn_heads = x_attn.view(B_, N, self.num_heads, C // self.num_heads)
        x_attn_gated = x_attn_heads * gate_score
        x_attn = x_attn_gated.reshape(B_, N, C)

        x = self.proj(x_attn)
        x = self.proj_drop(x)
        return x

class LGSBlock(nn.Module):
    """
    Local-Global Synergistic Block
    Branch A: Mamba2 (SSD) for Global Context with SS2D Scanning
    Branch B: Gated Window Attention for Local Texture
    Fusion: ECAB (Efficient Channel Attention)
    
    Supports ablation modes:
    - 'hybrid': Both branches (default)
    - 'mamba2_only': Only Mamba2 branch
    - 'gated_attn_only': Only Gated Window Attention branch
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 mamba_d_state=64, mamba_d_conv=4, mamba_expand=2, use_mhc=False,
                 backbone_mode='hybrid', use_ecab=True):
        super().__init__()
        self.dim = dim
        self.backbone_mode = backbone_mode
        self.norm1 = norm_layer(dim)
        
        # Branch A: Mamba2 (only if hybrid or mamba2_only)
        self.use_mamba = backbone_mode in ['hybrid', 'mamba2_only']
        if self.use_mamba and Mamba2 is not None:
            self.mamba = Mamba2(
                d_model=dim,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand
            )
        else:
            self.mamba = nn.Identity() if self.use_mamba else None

        # Branch B: Gated Window Attention (only if hybrid or gated_attn_only)
        self.use_attn = backbone_mode in ['hybrid', 'gated_attn_only']
        self.window_size = window_size
        self.shift_size = shift_size
        if self.use_attn:
            self.attn = GatedWindowAttention(
                dim, window_size=window_size, num_heads=num_heads,
                qkv_bias=True, qk_scale=None, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = None

        # Fusion & Mixing
        self.use_mhc = use_mhc
        if backbone_mode == 'hybrid':
            if use_mhc:
                self.mhc = ManifoldResConnection(dim, num_streams=3)  # input, mamba, attn
            else:
                self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)  # 1x1 conv to merge branches
        else:
            # Single branch mode - no fusion needed
            self.mhc = None
            self.fusion_conv = None
            
        # Channel attention
        if use_ecab:
            self.cab = ECAB(dim)
        else:
            # Standard CAB (SE-Block style) - simplified version
            self.cab = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // 4, dim, 1),
                nn.Sigmoid()
            )
        self.use_ecab = use_ecab
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        """
        Args:
            x: (2B, L, C) where first B are frame0, last B are frame1 features
               (batch-concatenated as in VFIMamba)
        """
        twoB, L, C = x.shape
        B = twoB // 2
        shortcut = x
        x_norm = self.norm1(x)
        
        x_mamba = None
        x_attn = None
        
        # --- Branch A: Mamba2 (Global) with VFIMamba-style Interleaved SS2D ---
        # Interleaves tokens from frame0 and frame1 into one sequence,
        # enabling cross-frame temporal interaction within the SSM's recurrent state.
        if self.use_mamba:
            with torch.amp.autocast('cuda', enabled=False):
                x_view = x_norm.float().view(twoB, H, W, C)  # (2B, H, W, C)
                x_scan = interleaved_scan(x_view)      # (4B, 2*H*W, C)
                x_mamba_out = self.mamba(x_scan)        # (4B, 2*H*W, C)
                x_mamba_img = interleaved_merge(x_mamba_out, B, H, W)  # (2B, H, W, C)
                x_mamba = x_mamba_img.flatten(1, 2).to(x_norm.dtype)   # (2B, L, C)
        
        # --- Branch B: Window Attention (Local) ---
        # Operates per-frame independently (2B batch)
        if self.use_attn:
            x_2d = x_norm.view(twoB, H, W, C)
            
            # Shift Window Mask Computation
            if self.shift_size > 0:
                img_mask = torch.zeros((1, H, W, 1), device=x.device)
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
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                
                shifted_x = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x_2d
                attn_mask = None

            # Partition
            x_windows = window_partition(shifted_x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
            
            # Gated Attention with Mask
            attn_windows = self.attn(x_windows, mask=attn_mask) 
            
            # Reverse Partition
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            
            # Reverse Shift
            if self.shift_size > 0:
                x_attn = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x_attn = shifted_x
                
            x_attn = x_attn.view(twoB, L, C)
        
        # --- Fusion / Mixing ---
        if self.backbone_mode == 'hybrid':
            if self.use_mhc:
                x_fused_seq = self.mhc([shortcut, x_mamba, x_attn])
                x_fused = x_fused_seq.view(twoB, H, W, C).permute(0, 3, 1, 2).contiguous()
            else:
                x_cat = torch.cat([x_mamba, x_attn], dim=-1)
                x_cat = x_cat.transpose(1, 2).view(twoB, 2*C, H, W)
                x_fused = self.fusion_conv(x_cat)
        elif self.backbone_mode == 'mamba2_only':
            x_fused = x_mamba.transpose(1, 2).view(twoB, C, H, W)
        else:  # gated_attn_only
            x_fused = x_attn.transpose(1, 2).view(twoB, C, H, W)
        
        # Apply channel attention
        if self.use_ecab:
            x_fused = self.cab(x_fused)
        else:
            # Standard CAB (SE-style)
            se_weight = self.cab(x_fused)
            x_fused = x_fused * se_weight
            
        x_fused = x_fused.flatten(2).transpose(1, 2)  # (B, L, C)
        
        if self.use_mhc and self.backbone_mode == 'hybrid':
            x = x_fused
        else:
            x = shortcut + self.drop_path(x_fused)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class BasicLayer(nn.Module):
    """ A basic Swin/Mamba layer for one stage """
    def __init__(self, dim, output_dim, depth, num_heads, window_size, mlp_ratio=4., drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 backbone_mode='hybrid', use_mhc=False, use_ecab=True):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            LGSBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                backbone_mode=backbone_mode,
                use_mhc=use_mhc,
                use_ecab=use_ecab
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
    Supports ablation modes: 'hybrid', 'mamba2_only', 'gated_attn_only'
    """
    def __init__(self, embed_dims=[32, 64, 128], depths=[2, 2, 2], num_heads=[2, 4, 8], 
                 window_sizes=[7, 7, 7], mlp_ratios=[4, 4, 4], drop_rate=0., drop_path_rate=0.1,
                 backbone_mode='hybrid', use_mhc=False, use_ecab=True):
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
            layer = BasicLayer(
                dim=embed_dims[i_layer],
                output_dim=embed_dims[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_sizes[i_layer],
                mlp_ratio=mlp_ratios[i_layer],
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=nn.LayerNorm,
                downsample=None,
                backbone_mode=backbone_mode,
                use_mhc=use_mhc,
                use_ecab=use_ecab
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
        """
        VFIMamba-style batch processing: concatenate frames along batch dim
        so the Mamba branch can perform interleaved cross-frame scanning.
        
        Input: img0, img1: (B, 3, H, W)
        Output: list of multi-scale features, each (B, C_i, H_i, W_i)
        """
        B_orig = img0.shape[0]
        # Batch-concat: (2B, 3, H, W) — frame0 in first half, frame1 in second half
        x = torch.cat([img0, img1], dim=0)  # (2B, 3, H, W)
        x = self.patch_embed(x)  # (2B, C, H, W)
        
        outs = []
        twoB, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (2B, L, C)
        
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            # Output features: (2B, C, H, W) -> sum frame0 and frame1 features -> (B, C, H, W)
            x_out = x.transpose(1, 2).view(twoB, -1, H, W)
            # Merge the two frames' features by summation (cross-frame info already mixed by Mamba)
            x_out_merged = x_out[:B_orig] + x_out[B_orig:]  # (B, C, H, W)
            outs.append(x_out_merged)
            
            if i < self.num_layers - 1:
                # Downsample for next stage — keep 2B batch structure
                x_down = self.downsamplers[i](x_out)
                _, _, H, W = x_down.shape
                x = x_down.flatten(2).transpose(1, 2)

        return outs  # List of [Scale1, Scale2, Scale3] features

    def init_mamba_from_attn(self):
        """
        MaTVLM-style initialization: transfer Attention Q/K/V weights to Mamba2 B/C/x
        for each LGSBlock that has both branches (hybrid mode).
        Call this AFTER model construction, BEFORE training.
        """
        if self.backbone_mode != 'hybrid':
            print("init_mamba_from_attn: skipped (not hybrid mode)")
            return
        count = 0
        for layer in self.layers:
            for blk in layer.blocks:
                if blk.use_mamba and blk.use_attn and hasattr(blk.mamba, 'in_proj'):
                    matvlm_init_mamba2(blk.mamba, blk.attn)
                    count += 1
        print(f"MaTVLM init: transferred weights for {count} LGS blocks")


def build_backbone(cfg):
    # Map config to arguments
    return HybridBackbone(
        embed_dims=cfg['embed_dims'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_sizes=cfg['window_sizes'],
        mlp_ratios=cfg['mlp_ratios'],
        backbone_mode=cfg.get('backbone_mode', 'hybrid'),
        use_mhc=cfg.get('use_mhc', False),
        use_ecab=cfg.get('use_ecab', True)
    )