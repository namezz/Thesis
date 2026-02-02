import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineNet(nn.Module):
    """
    Phase 1: Simple Reconstruction Head
    Takes multi-scale features and outputs the interpolated frame.
    """
    def __init__(self, c=32):
        super(RefineNet, self).__init__()
        
        # Assuming 3 scales from backbone:
        # Scale 0: H, W (c)
        # Scale 1: H/2, W/2 (2c)
        # Scale 2: H/4, W/4 (4c)
        
        # Simple UNet-like upsampling
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(c*4, c, 4, 2, 1) # c*2 (from up2) + c*2 (from scale 1) -> c
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(c*2, c, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c, 3, 3, 1, 1),
            nn.Sigmoid() # Output pixel values 0-1
        )

    def forward(self, feats):
        # feats = [c, 2c, 4c] corresponding to H, H/2, H/4
        # Note: In config.py we defined embed_dims as [F, 2F, 4F...]
        
        f0, f1, f2 = feats[0], feats[1], feats[2]
        
        # Up from f2 (H/4) to H/2
        x = self.up2(f2) 
        # Concat with f1 (H/2)
        if x.shape != f1.shape:
             x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear')
        x = torch.cat([x, f1], dim=1)
        
        # Up from H/2 to H
        x = self.up1(x)
        # Concat with f0 (H)
        if x.shape != f0.shape:
             x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear')
        x = torch.cat([x, f0], dim=1)
        
        # Final prediction
        out = self.conv_out(x)
        return out