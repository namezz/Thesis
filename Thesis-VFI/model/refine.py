import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class RefineNet(nn.Module):
    """
    U-Net based Refinement Network (inspired by RIFE FusionNet)
    Fuses multi-scale features from the backbone to generate the final frame.
    """
    def __init__(self, c=32):
        super(RefineNet, self).__init__()
        
        # Backbone provides features at 3 scales:
        # F0 (H, W, c), F1 (H/2, W/2, 2c), F2 (H/4, W/4, 4c)
        
        # Downsample path (Encoder) - Processing concatenated inputs if needed, 
        # but here we primarily fuse backbone features.
        
        # Upsample path (Decoder)
        # We start from the coarsest feature F2 (4c)
        
        self.up0 = deconv(4*c, 2*c) # H/4 -> H/2
        self.up1 = deconv(4*c, c)   # (2c from up0 + 2c from F1) -> H
        self.up2 = deconv(2*c, c)   # (c from up1 + c from F0) -> 2H (if we wanted super-res, but here we stay at H)
        
        # Since output is H,W, let's adjust.
        # F2(H/4) -> up -> H/2. Cat F1(H/2).
        # H/2 -> up -> H. Cat F0(H).
        
        self.conv_up0 = deconv(4*c, 2*c)
        self.conv_up1 = deconv(4*c, c)
        
        self.conv_last = nn.Conv2d(2*c, 4, 3, 1, 1) # Output: 3 (RGB residual) + 1 (Mask)

    def forward(self, feats):
        # feats: [f0(c), f1(2c), f2(4c)]
        f0, f1, f2 = feats[0], feats[1], feats[2]
        
        # Upsample F2
        x = self.conv_up0(f2) # (4c -> 2c, H/2)
        
        # Cat with F1
        if x.shape != f1.shape:
            x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear')
        x = torch.cat([x, f1], dim=1) # (4c)
        
        # Upsample
        x = self.conv_up1(x) # (4c -> c, H)
        
        # Cat with F0
        if x.shape != f0.shape:
            x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear')
        x = torch.cat([x, f0], dim=1) # (2c)
        
        # Final prediction
        out = self.conv_last(x)
        
        res = torch.sigmoid(out[:, :3]) * 2 - 1 # Range [-1, 1]
        mask = torch.sigmoid(out[:, 3:4])       # Range [0, 1]
        
        return res, mask