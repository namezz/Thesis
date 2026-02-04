from .backbone import build_backbone
from .flow import build_flow_estimator
from .refine import RefineNet
from .warplayer import warp
from config import MODEL_CONFIG
import torch.nn as nn
import torch
import torch.nn.functional as F

class ThesisModel(nn.Module):
    def __init__(self, cfg):
        super(ThesisModel, self).__init__()
        self.backbone = build_backbone(cfg)
        
        self.use_flow = MODEL_CONFIG.get('USE_FLOW', False)
        if self.use_flow:
            self.flow_estimator = build_flow_estimator()
        
        # Refine Head
        c = cfg['embed_dims'][0]
        self.refine = RefineNet(c=c)
        
    def forward(self, x, timestep=0.5):
        img0, img1 = x[:, :3], x[:, 3:6]
        
        if self.use_flow:
            # Phase 2: Flow Guidance
            # 1. Estimate flow
            flow = self.flow_estimator(img0, img1) # (B, 4, H, W) for bidirectional flow
            
            # 2. Warp inputs to intermediate time t
            f0 = flow[:, :2] * timestep
            f1 = flow[:, 2:4] * (1 - timestep)
            
            warped_img0 = warp(img0, f0)
            warped_img1 = warp(img1, f1)
            
            # 3. Extract features from warped images (Feature Pre-warping)
            # For Phase 2, we should probably feed warped images to backbone
            feats = self.backbone(warped_img0, warped_img1)
            
            # 4. Refine
            res, mask = self.refine(feats)
            
            # Blending
            merged = warped_img0 * mask + warped_img1 * (1 - mask)
            pred = merged + res
            return torch.clamp(pred, 0, 1)
        else:
            # Phase 1: Backbone only (Implicit Motion / Direct Regression)
            # In this phase, we rely on the backbone to find correspondences.
            # We treat the output as a refinement over a simple blend or direct prediction.
            feats = self.backbone(img0, img1)
            res, mask = self.refine(feats)
            
            # For Phase 1 baseline, simple linear blend as base
            merged = img0 * (1 - timestep) + img1 * timestep
            # Or use learned mask to blend original frames (Attention-based averaging)
            merged = img0 * mask + img1 * (1 - mask)
            
            pred = merged + res
            return torch.clamp(pred, 0, 1)

    def inference(self, img0, img1, TTA=False, scale=1.0, timestep=0.5, fast_TTA=False):
        # Scale handling
        if scale != 1.0:
            h, w = img0.shape[2], img0.shape[3]
            # Downsample for processing
            img0_s = F.interpolate(img0, scale_factor=scale, mode='bilinear', align_corners=False)
            img1_s = F.interpolate(img1, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            img0_s, img1_s = img0, img1

        # Padding
        B, C, H, W = img0_s.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h != 0 or pad_w != 0:
            img0_s = F.pad(img0_s, (0, pad_w, 0, pad_h), mode='reflect')
            img1_s = F.pad(img1_s, (0, pad_w, 0, pad_h), mode='reflect')
            
        def _infer(i0, i1):
            x = torch.cat((i0, i1), 1)
            return self.forward(x, timestep)

        if TTA:
            # Test Time Augmentation: Average of forward and flipped
            pred = _infer(img0_s, img1_s)
            
            # Flip inputs: Horizontal, Vertical, or Rotation?
            # Standard VFI TTA: Horizontal flip and Swap (if t=0.5)
            # RIFE/EMA-VFI use Horizontal Flip
            img0_flip = img0_s.flip(3)
            img1_flip = img1_s.flip(3)
            pred_flip = _infer(img0_flip, img1_flip)
            pred = (pred + pred_flip.flip(3)) / 2
        else:
            pred = _infer(img0_s, img1_s)
        
        # Unpad
        if pad_h != 0 or pad_w != 0:
            pred = pred[:, :, :H, :W]
            
        # Upsample back if scaled
        if scale != 1.0:
            pred = F.interpolate(pred, size=(img0.shape[2], img0.shape[3]), mode='bilinear', align_corners=False)
            
        return pred