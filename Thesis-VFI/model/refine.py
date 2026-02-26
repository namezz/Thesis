"""
refine.py — Multi-scale Refinement Network for VFI
====================================================
Generates the final interpolated frame by refining the flow-warped result.

Pipeline integration:
    img0, img1 → [Backbone] → feats
    feats → [FlowEstimator] → flow_01, flow_10, blend_mask
    warp(img0, flow_01), warp(img1, flow_10) → warped frames
    feats + warped_frames + blend_mask → [RefineNet] → RGB residual

Key improvements over original:
  1. Residual-on-warped: takes warped frames as input, learns correction only.
     This is much easier than generating pixels from scratch.
  2. Single responsibility: only outputs RGB residual (no mask — flow estimator
     handles blending mask). Eliminates mask conflict.
  3. PixelShuffle upsampling: avoids ConvTranspose2d checkerboard artifacts.
  4. Channel attention at each decoder level: re-weights features from backbone
     vs. warped context adaptively.
  5. Multi-scale output: predictions at each decoder level for multi-scale loss.
  6. Unbounded residual with tanh: better gradient flow for large corrections
     than sigmoid * 2 - 1.

References:
  - IFRNet (Kong et al., CVPR 2022): multi-scale decoder with intermediate supervision
  - EMA-VFI (Zhang et al., CVPR 2023): context-guided refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ════════════════════════════════════════════════════════════════════════════

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class ResBlock(nn.Module):
    """Residual block with PReLU."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        return self.prelu(self.block(x) + x)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Re-weights channels adaptively — critical at skip connection junctions
    where backbone features (semantic) meet warped features (appearance).
    The network learns which channels carry useful information at each scale.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class UpBlock(nn.Module):
    """
    PixelShuffle 2× upsampling + conv refinement.
    Replaces ConvTranspose2d to avoid checkerboard artifacts.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
            nn.PReLU(out_channels),
        )

    def forward(self, x):
        return self.up(x)


# ════════════════════════════════════════════════════════════════════════════
# Refinement Network
# ════════════════════════════════════════════════════════════════════════════

class RefineNet(nn.Module):
    """
    Multi-scale U-Net decoder for frame refinement.

    Consumes backbone multi-scale features and flow-warped context to produce
    a residual correction on the blended warped frame.

    Data flow:
        warped_blend = mask * warp(img0, f01) + (1-mask) * warp(img1, f10)
              ↓
        [RefineNet] processes backbone features + warped context
              ↓
        final_frame = warped_blend + residual  (clamped to [0, 1])

    Decoder structure (bottom-up):
        Level 2 (coarsest): feat_s2 [+ ctx_s2] → UpBlock → hidden_s1
        Level 1 (middle):   hidden_s1 + feat_s1 [+ ctx_s1] → UpBlock → hidden_s0
        Level 0 (finest):   hidden_s0 + feat_s0 [+ ctx_s0] + warped_blend → ResBlocks → residual

    Each level has:
      - Channel attention after concatenation (adaptive feature re-weighting)
      - PixelShuffle upsampling (no checkerboard)
      - Optional intermediate RGB prediction (for multi-scale loss)

    Args:
        embed_dims:      backbone feature dims per scale, e.g. [32, 64, 128]
        use_context:     Phase 2+: use warped context features from ContextNet
        multiscale_pred: output predictions at each scale (for multi-scale loss)
    """

    def __init__(self, embed_dims=[32, 64, 128], use_context=False,
                 multiscale_pred=True):
        super().__init__()
        self.use_context = use_context
        self.multiscale_pred = multiscale_pred
        self.num_scales = len(embed_dims)

        d0, d1, d2 = embed_dims[0], embed_dims[1], embed_dims[2]

        # Context channels per scale (warped feat0 + warped feat1 = 2× per scale)
        ctx_mult = 2 if use_context else 0

        # ---- Level 2 (coarsest): compress and start decoding ----
        in_ch_2 = d2 + d2 * ctx_mult
        self.compress_2 = nn.Sequential(
            conv(in_ch_2, d2),
            ResBlock(d2),
            ResBlock(d2),
        )
        self.ca_2 = ChannelAttention(d2)
        self.up_2 = UpBlock(d2, d1)

        # ---- Level 1 (middle): fuse with skip connection ----
        in_ch_1 = d1 + d1 + d1 * ctx_mult  # from up_2 + feat_s1 + ctx_s1
        self.compress_1 = nn.Sequential(
            conv(in_ch_1, d1),
            ResBlock(d1),
        )
        self.ca_1 = ChannelAttention(d1)
        self.up_1 = UpBlock(d1, d0)

        # ---- Level 0 (finest): fuse with skip + warped blend ----
        # Input: from up_1 + feat_s0 + ctx_s0 + warped_blend(3ch)
        in_ch_0 = d0 + d0 + d0 * ctx_mult + 3
        self.compress_0 = nn.Sequential(
            conv(in_ch_0, d0),
            ResBlock(d0),
            ResBlock(d0),
        )
        self.ca_0 = ChannelAttention(d0)

        # ---- Output head: RGB residual only (no mask) ----
        self.output_head = nn.Sequential(
            conv(d0, d0),
            nn.Conv2d(d0, 3, 3, 1, 1, bias=True),
            nn.Tanh(),  # residual in [-1, 1], better gradient than sigmoid
        )

        # ---- Multi-scale prediction heads (optional) ----
        if multiscale_pred:
            self.pred_head_2 = nn.Sequential(
                conv(d2, d2 // 2),
                nn.Conv2d(d2 // 2, 3, 1, bias=True),
                nn.Tanh(),
            )
            self.pred_head_1 = nn.Sequential(
                conv(d1, d1 // 2),
                nn.Conv2d(d1 // 2, 3, 1, bias=True),
                nn.Tanh(),
            )

    def forward(self, feats, warped_blend, ctx0=None, ctx1=None):
        """
        Args:
            feats:         [feat_s0, feat_s1, feat_s2] from backbone
                           Each is (B, C_i, H_i, W_i)
            warped_blend:  (B, 3, H, W) — mask-blended warped frame
                           = mask * warp(img0, f01) + (1-mask) * warp(img1, f10)
            ctx0:          [c0_s0, c0_s1, c0_s2] warped context of img0 (Phase 2+)
            ctx1:          [c1_s0, c1_s1, c1_s2] warped context of img1 (Phase 2+)

        Returns:
            residual:      (B, 3, H, W) — RGB correction in [-1, 1]
                           final_frame = clamp(warped_blend + residual, 0, 1)
            pred_list:     list of multi-scale predictions (for multi-scale loss)
                           Each is (B, 3, H_i, W_i), already clamped to [0, 1]
        """
        f0, f1, f2 = feats[0], feats[1], feats[2]
        pred_list = []

        # ---- Level 2 (coarsest) ----
        cat_2 = [f2]
        if self.use_context and ctx0 is not None and ctx1 is not None:
            cat_2.extend([ctx0[2], ctx1[2]])
        x = self.compress_2(torch.cat(cat_2, dim=1))
        x = self.ca_2(x)

        if self.multiscale_pred:
            res_2 = self.pred_head_2(x)
            wb_2 = F.interpolate(warped_blend, size=res_2.shape[-2:],
                                 mode='bilinear', align_corners=False)
            pred_2 = torch.clamp(wb_2 + res_2, 0, 1)
            pred_list.append(pred_2)

        x = self.up_2(x)

        # ---- Level 1 (middle) ----
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:],
                              mode='bilinear', align_corners=False)
        cat_1 = [x, f1]
        if self.use_context and ctx0 is not None and ctx1 is not None:
            cat_1.extend([ctx0[1], ctx1[1]])
        x = self.compress_1(torch.cat(cat_1, dim=1))
        x = self.ca_1(x)

        if self.multiscale_pred:
            res_1 = self.pred_head_1(x)
            wb_1 = F.interpolate(warped_blend, size=res_1.shape[-2:],
                                 mode='bilinear', align_corners=False)
            pred_1 = torch.clamp(wb_1 + res_1, 0, 1)
            pred_list.append(pred_1)

        x = self.up_1(x)

        # ---- Level 0 (finest) ----
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:],
                              mode='bilinear', align_corners=False)
        wb_0 = warped_blend
        if wb_0.shape[-2:] != f0.shape[-2:]:
            wb_0 = F.interpolate(warped_blend, size=f0.shape[-2:],
                                 mode='bilinear', align_corners=False)
        cat_0 = [x, f0, wb_0]
        if self.use_context and ctx0 is not None and ctx1 is not None:
            cat_0.extend([ctx0[0], ctx1[0]])
        x = self.compress_0(torch.cat(cat_0, dim=1))
        x = self.ca_0(x)

        # ---- Final output ----
        residual = self.output_head(x)

        if residual.shape[-2:] != warped_blend.shape[-2:]:
            residual = F.interpolate(residual, size=warped_blend.shape[-2:],
                                     mode='bilinear', align_corners=False)
        pred_full = torch.clamp(warped_blend + residual, 0, 1)
        pred_list.append(pred_full)

        # Reverse to [finest, ..., coarsest] to match multiscale_weights convention
        pred_list = list(reversed(pred_list))

        return residual, pred_list


# ════════════════════════════════════════════════════════════════════════════
# Builder
# ════════════════════════════════════════════════════════════════════════════

def build_refine_net(cfg=None, embed_dims=None, use_context=False,
                     multiscale_pred=True):
    """
    Build refinement network.

    Args:
        cfg:             config dict with 'embed_dims' key
        embed_dims:      list of backbone embed dims (overrides cfg)
        use_context:     enable context feature input (Phase 2+)
        multiscale_pred: enable multi-scale predictions for loss

    Example:
        refine = build_refine_net(embed_dims=[32, 64, 128])

        # Phase 1 (no context):
        residual, preds = refine(backbone_feats, warped_blend)

        # Phase 2 (with context):
        refine = build_refine_net(embed_dims=[32, 64, 128], use_context=True)
        residual, preds = refine(backbone_feats, warped_blend, ctx0, ctx1)
    """
    if embed_dims is None and cfg is not None:
        embed_dims = cfg['embed_dims']
    if embed_dims is None:
        embed_dims = [32, 64, 128]

    return RefineNet(
        embed_dims=embed_dims,
        use_context=use_context,
        multiscale_pred=multiscale_pred,
    )