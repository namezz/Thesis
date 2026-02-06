from .backbone import build_backbone
from .flow import build_flow_estimator
from .refine import RefineNet
from .warplayer import warp
import torch.nn as nn
import torch
import torch.nn.functional as F

class ContextNet(nn.Module):
    """
    Multi-scale context feature extractor for warping (inspired by RIFE/VFIMamba).
    Extracts per-frame features at backbone scales, then warps them using flow.
    This provides the RefineNet with aligned contextual information.
    """
    def __init__(self, c=32):
        super(ContextNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, c, 3, 1, 1), nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1), nn.PReLU(c),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 2*c, 3, 2, 1), nn.PReLU(2*c),
            nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.PReLU(2*c),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*c, 4*c, 3, 2, 1), nn.PReLU(4*c),
            nn.Conv2d(4*c, 4*c, 3, 1, 1), nn.PReLU(4*c),
        )
    
    def forward(self, x, flow):
        """
        Extract multi-scale context and warp each scale by flow.
        Args:
            x: (B, 3, H, W) input image
            flow: (B, 2, H, W) optical flow for this image
        Returns:
            list of warped features at 3 scales
        """
        feat0 = self.conv0(x)     # (B, c, H, W)
        feat1 = self.conv1(feat0) # (B, 2c, H/2, W/2)
        feat2 = self.conv2(feat1) # (B, 4c, H/4, W/4)
        
        # Warp each scale using appropriately downscaled flow
        w_feat0 = warp(feat0, flow)
        flow_2 = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
        w_feat1 = warp(feat1, flow_2)
        flow_4 = F.interpolate(flow, scale_factor=0.25, mode='bilinear', align_corners=False) * 0.25
        w_feat2 = warp(feat2, flow_4)
        
        return [w_feat0, w_feat1, w_feat2]

class ThesisModel(nn.Module):
    def __init__(self, cfg):
        super(ThesisModel, self).__init__()
        self.backbone = build_backbone(cfg)
        
        self.use_flow = cfg.get('use_flow', False)
        if self.use_flow:
            self.flow_estimator = build_flow_estimator()
            c = cfg['embed_dims'][0]
            # Context nets extract per-frame features to warp with flow
            self.context0 = ContextNet(c=c)
            self.context1 = ContextNet(c=c)
        
        # Refine Head
        c = cfg['embed_dims'][0]
        self.refine = RefineNet(c=c, use_context=self.use_flow)
        
    def forward(self, x, timestep=0.5):
        img0, img1 = x[:, :3], x[:, 3:6]
        
        if self.use_flow:
            # Phase 2: Flow-guided feature warping
            flow, flow_mask = self.flow_estimator(img0, img1, timestep=timestep)
            
            # Flow is bidirectional: flow[:, :2] = img0→t, flow[:, 2:4] = img1→t
            f0 = flow[:, :2]
            f1 = flow[:, 2:4]
            
            # Warp images for blending
            warped_img0 = warp(img0, f0)
            warped_img1 = warp(img1, f1)
            
            # Extract and warp context features (RIFE/VFIMamba style)
            ctx0 = self.context0(img0, f0)
            ctx1 = self.context1(img1, f1)
            
            # Backbone processes ORIGINAL frames (cross-frame attention on unwarped content)
            feats = self.backbone(img0, img1)
            
            # RefineNet receives backbone features + warped context
            res, refine_mask = self.refine(feats, ctx0=ctx0, ctx1=ctx1)
            
            # Combine flow mask and refine mask
            mask = torch.sigmoid(flow_mask + refine_mask)
            merged = warped_img0 * mask + warped_img1 * (1 - mask)
            pred = merged + res
            pred = torch.clamp(pred, 0, 1)
            return pred, flow
        else:
            # Phase 1: Backbone only (Direct Regression)
            feats = self.backbone(img0, img1)
            res, mask = self.refine(feats)
            
            # Learned mask blending of original frames + residual refinement
            merged = img0 * mask + img1 * (1 - mask)
            pred = merged + res
            pred = torch.clamp(pred, 0, 1)
            return pred, None

    def inference(self, img0, img1, TTA=False, scale=1.0, timestep=0.5, fast_TTA=False):
        """
        Inference with Test Time Augmentation (TTA) support.
        
        TTA Modes:
        - TTA=False: Standard forward pass
        - TTA=True, fast_TTA=False: Full TTA (H-flip + V-flip + Transpose) = 8 augmentations
        - TTA=True, fast_TTA=True: Fast TTA (H-flip only) = 2 augmentations
        
        Reference: RIFE, VFIMamba use H-flip for TTA
        """
        # Scale handling for high-resolution
        if scale != 1.0:
            img0_s = F.interpolate(img0, scale_factor=scale, mode='bilinear', align_corners=False)
            img1_s = F.interpolate(img1, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            img0_s, img1_s = img0, img1

        # Padding to ensure divisibility by 32
        B, C, H, W = img0_s.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h != 0 or pad_w != 0:
            img0_s = F.pad(img0_s, (0, pad_w, 0, pad_h), mode='reflect')
            img1_s = F.pad(img1_s, (0, pad_w, 0, pad_h), mode='reflect')
            
        def _infer(i0, i1):
            x = torch.cat((i0, i1), 1)
            pred, _ = self.forward(x, timestep)
            return pred

        if TTA:
            if fast_TTA:
                pred_normal = _infer(img0_s, img1_s)
                pred_hflip = _infer(img0_s.flip(3), img1_s.flip(3)).flip(3)
                pred = (pred_normal + pred_hflip) / 2
            else:
                pred = _infer(img0_s, img1_s)
                pred += _infer(img0_s.flip(3), img1_s.flip(3)).flip(3)
                pred += _infer(img0_s.flip(2), img1_s.flip(2)).flip(2)
                pred += _infer(img0_s.flip(2, 3), img1_s.flip(2, 3)).flip(2, 3)
                img0_t = img0_s.transpose(2, 3)
                img1_t = img1_s.transpose(2, 3)
                pred += _infer(img0_t, img1_t).transpose(2, 3)
                pred += _infer(img0_t.flip(3), img1_t.flip(3)).flip(3).transpose(2, 3)
                pred += _infer(img0_t.flip(2), img1_t.flip(2)).flip(2).transpose(2, 3)
                pred += _infer(img0_t.flip(2, 3), img1_t.flip(2, 3)).flip(2, 3).transpose(2, 3)
                pred = pred / 8
        else:
            pred = _infer(img0_s, img1_s)
        
        # Unpad
        if pad_h != 0 or pad_w != 0:
            pred = pred[:, :, :H, :W]
            
        # Upsample back if scaled
        if scale != 1.0:
            pred = F.interpolate(pred, size=(img0.shape[2], img0.shape[3]), mode='bilinear', align_corners=False)
            
        return pred