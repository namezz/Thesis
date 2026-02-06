from .backbone import build_backbone
from .flow import build_flow_estimator
from .refine import RefineNet
from .warplayer import warp
import torch.nn as nn
import torch
import torch.nn.functional as F

class ThesisModel(nn.Module):
    def __init__(self, cfg):
        super(ThesisModel, self).__init__()
        self.backbone = build_backbone(cfg)
        
        self.use_flow = cfg.get('use_flow', False)
        if self.use_flow:
            self.flow_estimator = build_flow_estimator()
        
        # Refine Head
        c = cfg['embed_dims'][0]
        self.refine = RefineNet(c=c)
        
    def forward(self, x, timestep=0.5):
        img0, img1 = x[:, :3], x[:, 3:6]
        
        if self.use_flow:
            # Phase 2: Flow Guidance
            flow, flow_mask = self.flow_estimator(img0, img1)
            
            f0 = flow[:, :2] * timestep
            f1 = flow[:, 2:4] * (1 - timestep)
            
            warped_img0 = warp(img0, f0)
            warped_img1 = warp(img1, f1)
            
            feats = self.backbone(warped_img0, warped_img1)
            res, refine_mask = self.refine(feats)
            
            # Combine flow mask and refine mask
            mask = torch.sigmoid(flow_mask + refine_mask)
            merged = warped_img0 * mask + warped_img1 * (1 - mask)
            pred = merged + res
            pred = torch.clamp(pred, 0, 1)
            # Return flow for flow smoothness loss
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
            h, w = img0.shape[2], img0.shape[3]
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
                # Fast TTA: Horizontal flip only (2x)
                pred_normal = _infer(img0_s, img1_s)
                pred_hflip = _infer(img0_s.flip(3), img1_s.flip(3)).flip(3)
                pred = (pred_normal + pred_hflip) / 2
            else:
                # Full TTA: 8 augmentations (H-flip, V-flip, Transpose)
                # Aug 0: Original
                pred = _infer(img0_s, img1_s)
                
                # Aug 1: Horizontal flip
                pred += _infer(img0_s.flip(3), img1_s.flip(3)).flip(3)
                
                # Aug 2: Vertical flip
                pred += _infer(img0_s.flip(2), img1_s.flip(2)).flip(2)
                
                # Aug 3: H+V flip (Rotate 180)
                pred += _infer(img0_s.flip(2, 3), img1_s.flip(2, 3)).flip(2, 3)
                
                # Aug 4-7: Transpose + flips
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