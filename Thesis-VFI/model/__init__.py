from .backbone import build_backbone
from .flow import build_flow_estimator
from .refine import RefineNet
import torch.nn as nn
import torch

class ThesisModel(nn.Module):
    def __init__(self, cfg):
        super(ThesisModel, self).__init__()
        self.backbone = build_backbone(cfg)
        self.flow_estimator = build_flow_estimator()
        self.refine = RefineNet()
        
    def forward(self, x):
        img0, img1 = x[:, :3], x[:, 3:6]
        
        # Phase 1: Backbone only (conceptually)
        feat = self.backbone(img0, img1)
        
        # Phase 2: Flow Guidance (Placeholder logic)
        flow = self.flow_estimator(img0, img1)
        
        # Refine to output
        res = self.refine(feat)
        return res # Placeholder: Needs proper merging of warped images + residuals

    def inference(self, img0, img1):
        # Inference wrapper
        return self.forward(torch.cat((img0, img1), 1))
