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
            # Alternatively, extract then warp. Let's start with image-level warping for baseline.
            feats = self.backbone(warped_img0, warped_img1)
            
            # 4. Refine with warped info
            # Refine head will need to handle warped images eventually.
            res = self.refine(feats)
            
            # Simple blending for now
            mask = 0.5 # Placeholder for learned mask
            merged = warped_img0 * (1-mask) + warped_img1 * mask
            return torch.clamp(merged + (res * 2 - 1) * 0.1, 0, 1)
        else:
            # Phase 1: Backbone only (Implicit Motion)
            feats = self.backbone(img0, img1)
            res = self.refine(feats)
            return res

    def inference(self, img0, img1, timestep=0.5):
        # Inference wrapper
        # Ensure input is multiple of window size or pad it?
        # The Backbone handles arbitrary sizes but Window Attention needs padding usually.
        # For simplicity in Phase 1, we assume inputs are divisible or handle errors.
        # Adding simple padding logic here is good practice.
        
        B, C, H, W = img0.shape
        # Pad to multiple of 32 (standard for 5-level models, here 3 levels -> 8 is enough, but 32 is safe)
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h != 0 or pad_w != 0:
            img0 = F.pad(img0, (0, pad_w, 0, pad_h), mode='reflect')
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='reflect')
            
        x = torch.cat((img0, img1), 1)
        pred = self.forward(x)
        
        if pad_h != 0 or pad_w != 0:
            pred = pred[:, :, :H, :W]
            
        return pred