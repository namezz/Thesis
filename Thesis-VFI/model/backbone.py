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

class GatedWindowAttention(nn.Module):
    """
    Window based multi-head self-attention (W-MSA) with gating mechanism.
    Ref: Gated Attention (arXiv:2505.06708, Qiu et al., Qwen Team, 2025)
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.gate = nn.Linear(dim, num_heads) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = mask
        if mask is not None:
            nW = mask.shape[0]
            attn_mask = mask.unsqueeze(1).unsqueeze(0)
            attn_mask = attn_mask.repeat(B_ // nW, 1, 1, 1, 1)
            attn_mask = attn_mask.view(-1, 1, N, N)

        x_attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            dropout_p=self.attn_drop.p if self.training else 0., 
            scale=self.scale
        )
        
        x_attn = x_attn.transpose(1, 2).reshape(B_, N, C)
        gate_score = self.sigmoid(self.gate(x)).unsqueeze(-1) 
        x_attn_heads = x_attn.view(B_, N, self.num_heads, C // self.num_heads)
        x_attn_gated = (x_attn_heads * gate_score).reshape(B_, N, C)

        x = self.proj(x_attn_gated)
        x = self.proj_drop(x)
        return x

# ═══════════════════════════════════════════════════════════════════════════
# Nested S-shaped Scan (NSS) — MaIR CVPR 2025
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=32)
def _build_nss_indices(H: int, W: int, stripe_width: int,
                       direction: str) -> torch.Tensor:
    L = H * W
    indices = torch.zeros(L, dtype=torch.long)
    if direction.startswith('h'):
        pos = 0
        num_stripes = (H + stripe_width - 1) // stripe_width
        for s in range(num_stripes):
            row_start = s * stripe_width
            row_end = min(row_start + stripe_width, H)
            for local_r, r in enumerate(range(row_start, row_end)):
                if local_r % 2 == 0:
                    for c in range(W):
                        indices[pos] = r * W + c
                        pos += 1
                else:
                    for c in range(W - 1, -1, -1):
                        indices[pos] = r * W + c
                        pos += 1
        if direction == 'h_bwd': indices = indices.flip(0)
    else:
        pos = 0
        num_stripes = (W + stripe_width - 1) // stripe_width
        for s in range(num_stripes):
            col_start = s * stripe_width
            col_end = min(col_start + stripe_width, W)
            for local_c, c in enumerate(range(col_start, col_end)):
                if local_c % 2 == 0:
                    for r in range(H):
                        indices[pos] = r * W + c
                        pos += 1
                else:
                    for r in range(H - 1, -1, -1):
                        indices[pos] = r * W + c
                        pos += 1
        if direction == 'v_bwd': indices = indices.flip(0)
    return indices

@lru_cache(maxsize=32)
def _build_nss_inverse(H: int, W: int, stripe_width: int,
                       direction: str) -> torch.Tensor:
    fwd_idx = _build_nss_indices(H, W, stripe_width, direction)
    inv_idx = torch.empty_like(fwd_idx)
    inv_idx[fwd_idx] = torch.arange(H * W)
    return inv_idx

class NSScan(nn.Module):
    def __init__(self, stripe_width: int = 4, num_directions: int = 4, shift: bool = False):
        super().__init__()
        self.stripe_width = stripe_width
        self.num_directions = num_directions
        self.shift_offset = stripe_width // 2 if shift else 0
        if num_directions == 2: self.directions = ['h_fwd', 'v_fwd']
        else: self.directions = ['h_fwd', 'h_bwd', 'v_fwd', 'v_bwd']

    def scan(self, x_2d: torch.Tensor):
        N, H, W, C = x_2d.shape
        L = H * W
        if self.shift_offset > 0:
            x_2d = torch.roll(x_2d, shifts=(-self.shift_offset, -self.shift_offset), dims=(1, 2))
        x_flat = x_2d.reshape(N, L, C)
        seqs = [x_flat[:, _build_nss_indices(H, W, self.stripe_width, d).to(x_2d.device), :] for d in self.directions]
        return torch.cat(seqs, dim=0), {'H': H, 'W': W, 'N': N, 'C': C}

    def unscan(self, x_seqs: torch.Tensor, scan_info: dict):
        H, W, N, C = scan_info['H'], scan_info['W'], scan_info['N'], scan_info['C']
        L, num_dirs = H * W, self.num_directions
        chunks = x_seqs.chunk(num_dirs, dim=0)
        x_acc = torch.zeros(N, L, C, device=x_seqs.device, dtype=x_seqs.dtype)
        for seq, d in zip(chunks, self.directions):
            x_acc += seq[:, _build_nss_inverse(H, W, self.stripe_width, d).to(x_seqs.device), :]
        x_2d = (x_acc / num_dirs).reshape(N, H, W, C)
        if self.shift_offset > 0:
            x_2d = torch.roll(x_2d, shifts=(self.shift_offset, self.shift_offset), dims=(1, 2))
        return x_2d

class FactorizedSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, num_scan_dirs=4, stripe_width=4, shift=False, use_checkpointing=True, headdim=64):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.spatial_mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.nss = NSScan(stripe_width=stripe_width, num_directions=num_scan_dirs, shift=shift)
        self.time_mix_proj = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.norm_s, self.norm_t = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x0_seq, x1_seq, H, W):
        B, L, C = x0_seq.shape
        x_cat = torch.cat([x0_seq, x1_seq], dim=0)
        res = x_cat
        with torch.amp.autocast('cuda', enabled=False):
            x_norm = self.norm_s(x_cat.float())
            x_scanned, info = self.nss.scan(x_norm.view(2 * B, H, W, C))
            if self.training and self.use_checkpointing:
                out = grad_checkpoint(self.spatial_mamba, x_scanned, use_reentrant=False)
            else:
                out = self.spatial_mamba(x_scanned)
            spatial_out = self.nss.unscan(out, info).reshape(2 * B, L, C) + res.float()
        spatial_out = spatial_out.to(x0_seq.dtype)
        f0, f1 = spatial_out[:B], spatial_out[B:]
        f0_f = self.norm_t(f0 + self.time_mix_proj(torch.cat([f0, f1], dim=-1)))
        f1_f = self.norm_t(f1 + self.time_mix_proj(torch.cat([f1, f0], dim=-1)))
        return f0_f, f1_f

class LGSBlock(nn.Module):
    """
    Local-Global Synergistic Block.
    Upgraded with Channel Split (Feature Shunting) for extreme efficiency.
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mamba_d_state=64, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64, num_scan_dirs=4, stripe_width=4, backbone_mode='hybrid', use_ecab=True, use_checkpointing=True):
        super().__init__()
        self.dim, self.backbone_mode, self.norm1 = dim, backbone_mode, norm_layer(dim)
        
        # Channel Split logic: half to Mamba, half to Attention
        self.half_dim = dim // 2 if backbone_mode == 'hybrid' else dim
        
        # Mamba2 requires d_ssm (half_dim * expand) to be divisible by headdim.
        # Standard expand=2, so d_ssm = dim. 
        # For Ultra (dim=64, half=32), d_ssm=64. headdim=64 is fine.
        # But if headdim > d_ssm, it will fail.
        actual_headdim = min(mamba_headdim, self.half_dim * mamba_expand)
        
        self.use_mamba = backbone_mode in ['hybrid', 'mamba2_only']
        if self.use_mamba:
            self.factorized_ssm = FactorizedSSMBlock(self.half_dim, mamba_d_state, mamba_d_conv, mamba_expand, num_scan_dirs, stripe_width, shift_size > 0, use_checkpointing, actual_headdim)
        
        self.use_attn = backbone_mode in ['hybrid', 'gated_attn_only']
        self.window_size, self.shift_size = window_size, shift_size
        if self.use_attn:
            self.attn = GatedWindowAttention(self.half_dim, window_size, num_heads, True, None, attn_drop, drop)
        
        # Fusion and Post-processing
        if backbone_mode == 'hybrid':
            self.cross_gate = CrossGatingFusion(self.half_dim)
            # Projection to restore full dim if we used splitting
            self.out_proj = nn.Conv2d(self.half_dim, dim, kernel_size=1)
        
        self.cab = ECAB(dim) if use_ecab else nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim // 4, 1), nn.ReLU(True), nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid())
        self.use_ecab = use_ecab
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2, self.mlp = norm_layer(dim), Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def _apply_cab(self, x_2d):
        if self.use_ecab: return self.cab(x_2d)
        else: return x_2d * self.cab(x_2d)

    def forward(self, x, H, W):
        twoB, L, C = x.shape
        B, shortcut, x_norm = twoB // 2, x, self.norm1(x)
        
        if self.backbone_mode == 'hybrid':
            # === Feature Shunting (Channel Split) ===
            x_mamba_in = x_norm[..., :self.half_dim]
            x_attn_in = x_norm[..., self.half_dim:]
            
            # Mamba Branch
            f0_m, f1_m = self.factorized_ssm(x_mamba_in[:B], x_mamba_in[B:], H, W)
            xm = torch.cat([f0_m, f1_m], dim=0) # [2B, L, C/2]
            
            # Attention Branch
            xa_2d = x_attn_in.view(twoB, H, W, self.half_dim)
            if self.shift_size > 0:
                img_mask = torch.zeros((1, H, W, 1), device=x.device)
                h_s = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                w_s = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                cnt = 0
                for h in h_s:
                    for w in w_s:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                m_w = window_partition(img_mask, self.window_size).view(-1, self.window_size**2)
                a_m = (m_w.unsqueeze(1) - m_w.unsqueeze(2)).masked_fill(m_w.unsqueeze(1)-m_w.unsqueeze(2) != 0, -100.0).masked_fill(m_w.unsqueeze(1)-m_w.unsqueeze(2) == 0, 0.0)
                s_x = torch.roll(xa_2d, (-self.shift_size, -self.shift_size), (1, 2))
            else: s_x, a_m = xa_2d, None
            w_x = window_partition(s_x, self.window_size).view(-1, self.window_size**2, self.half_dim)
            a_w = self.attn(w_x, a_m).view(-1, self.window_size, self.window_size, self.half_dim)
            r_x = window_reverse(a_w, self.window_size, H, W)
            xa = torch.roll(r_x, (self.shift_size, self.shift_size), (1, 2)).view(twoB, L, self.half_dim) if self.shift_size > 0 else r_x.view(twoB, L, self.half_dim)
            
            # Fusion
            fm_2d = xm.transpose(1, 2).view(twoB, self.half_dim, H, W)
            fa_2d = xa.transpose(1, 2).view(twoB, self.half_dim, H, W)
            x_fused = self.cross_gate(fm_2d, fa_2d) # [2B, C/2, H, W]
            
            # Project back to full dim and APPLY ECAB (Fixing the bug!)
            x_fused = self.out_proj(x_fused)
            x_fused = self._apply_cab(x_fused)
            
            x_fused = x_fused.flatten(2).transpose(1, 2)
            x = shortcut + self.drop_path(x_fused)
            
        else:
            # Single branch modes (No splitting)
            if self.use_mamba:
                f0_m, f1_m = self.factorized_ssm(x_norm[:B], x_norm[B:], H, W)
                x_f = torch.cat([f0_m, f1_m], dim=0)
            else:
                x_2d = x_norm.view(twoB, H, W, C)
                # ... [Keep original attention logic for single branch] ...
                if self.shift_size > 0:
                    img_mask = torch.zeros((1, H, W, 1), device=x.device)
                    h_s = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                    w_s = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
                    cnt = 0
                    for h in h_s:
                        for w in w_s:
                            img_mask[:, h, w, :] = cnt
                            cnt += 1
                    m_w = window_partition(img_mask, self.window_size).view(-1, self.window_size**2)
                    a_m = (m_w.unsqueeze(1) - m_w.unsqueeze(2)).masked_fill(m_w.unsqueeze(1)-m_w.unsqueeze(2) != 0, -100.0).masked_fill(m_w.unsqueeze(1)-m_w.unsqueeze(2) == 0, 0.0)
                    s_x = torch.roll(x_2d, (-self.shift_size, -self.shift_size), (1, 2))
                else: s_x, a_m = x_2d, None
                w_x = window_partition(s_x, self.window_size).view(-1, self.window_size**2, C)
                a_w = self.attn(w_x, a_m).view(-1, self.window_size, self.window_size, C)
                r_x = window_reverse(a_w, self.window_size, H, W)
                x_f = torch.roll(r_x, (self.shift_size, self.shift_size), (1, 2)).view(twoB, L, C) if self.shift_size > 0 else r_x.view(twoB, L, C)
            
            x_f_2d = x_f.transpose(1, 2).view(twoB, C, H, W)
            x_f_2d = self._apply_cab(x_f_2d)
            x = shortcut + self.drop_path(x_f_2d.flatten(2).transpose(1, 2))
            
        return x + self.drop_path(self.mlp(self.norm2(x)))

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, num_scan_dirs=4, stripe_width=4, backbone_mode='hybrid', use_ecab=True, use_checkpointing=True, mamba_headdim=64):
        super().__init__()
        self.blocks = nn.ModuleList([LGSBlock(dim, num_heads, window_size, 0 if i % 2 == 0 else window_size // 2, mlp_ratio, drop, 0, drop_path[i] if isinstance(drop_path, list) else drop_path, nn.GELU, norm_layer, 64, 4, 2, mamba_headdim, num_scan_dirs, stripe_width, backbone_mode, use_ecab, use_checkpointing) for i in range(depth)])
    def forward(self, x, H, W):
        for b in self.blocks: x = b(x, H, W)
        return x, H, W

class HybridBackbone(nn.Module):
    def __init__(self, embed_dims=[32, 64, 128], depths=[2, 2, 2], num_heads=[2, 4, 8], window_sizes=[8, 8, 8], mlp_ratios=[4, 4, 4], drop_rate=0., drop_path_rate=0.1, num_scan_dirs=4, stripe_width=4, backbone_mode='hybrid', use_ecab=True, use_checkpointing=True, mamba_headdim=64):
        super().__init__()
        self.num_layers, self.embed_dims, self.backbone_mode = len(depths), embed_dims, backbone_mode
        self.patch_embed = nn.Sequential(nn.Conv2d(3, embed_dims[0], 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1), nn.LeakyReLU(0.1, True))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([BasicLayer(embed_dims[i], depths[i], num_heads[i], window_sizes[i], mlp_ratios[i], drop_rate, dpr[sum(depths[:i]):sum(depths[:i+1])], nn.LayerNorm, num_scan_dirs, stripe_width, backbone_mode, use_ecab, use_checkpointing, mamba_headdim) for i in range(self.num_layers)])
        self.downsamplers = nn.ModuleList([nn.Sequential(nn.Conv2d(embed_dims[i], embed_dims[i+1], 3, 2, 1), nn.LeakyReLU(0.1)) for i in range(self.num_layers - 1)])
        self.merge_convs = nn.ModuleList([nn.Conv2d(embed_dims[i]*2, embed_dims[i], 1) for i in range(self.num_layers)])

    def forward(self, img0, img1):
        B_orig = img0.shape[0]
        x = self.patch_embed(torch.cat([img0, img1], 0))
        outs, twoB, _, H, W = [], *x.shape
        x = x.flatten(2).transpose(1, 2)
        for i, l in enumerate(self.layers):
            x, H, W = l(x, H, W)
            x_out = x.transpose(1, 2).view(twoB, -1, H, W)
            outs.append(self.merge_convs[i](torch.cat([x_out[:B_orig], x_out[B_orig:]], 1)))
            if i < self.num_layers - 1:
                x_down = self.downsamplers[i](x_out)
                _, _, H, W = x_down.shape
                x = x_down.flatten(2).transpose(1, 2)
        return outs

    def load_legacy_weights(self, state_dict):
        own = self.state_dict()
        l, s = 0, 0
        for n, p in state_dict.items():
            if n in own and own[n].shape == p.shape:
                own[n].copy_(p); l += 1
            else: s += 1
        return l, s

def build_backbone(cfg):
    return HybridBackbone(cfg['embed_dims'], cfg['depths'], cfg['num_heads'], cfg['window_sizes'], cfg['mlp_ratios'], 0., 0.1, cfg.get('num_scan_dirs', 4), cfg.get('stripe_width', 4), cfg.get('backbone_mode', 'hybrid'), cfg.get('use_ecab', True), cfg.get('use_checkpointing', True), cfg.get('mamba_headdim', 64))
