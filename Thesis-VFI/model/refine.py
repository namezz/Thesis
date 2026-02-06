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
    U-Net based Refinement Network (inspired by RIFE/VFIMamba).
    Fuses multi-scale features from the backbone to generate the final frame.
    
    Phase 1: backbone features only
    Phase 2: backbone features + warped context features from ContextNet
    """
    def __init__(self, c=32, use_context=False):
        super(RefineNet, self).__init__()
        self.use_context = use_context
        
        # Backbone provides features at 3 scales:
        # F0 (H, W, c), F1 (H/2, W/2, 2c), F2 (H/4, W/4, 4c)
        
        # Phase 2: context features add c + 2c + 4c channels at each scale
        ctx_c = [c, 2*c, 4*c] if use_context else [0, 0, 0]
        
        # Decoder: bottom-up with skip connections
        # F2 (4c + ctx2*2) → 2c at H/2
        self.conv_up0 = deconv(4*c + ctx_c[2]*2, 2*c)
        # cat with F1 (2c + ctx1*2) → c at H
        self.conv_up1 = deconv(2*c + 2*c + ctx_c[1]*2, c)
        # cat with F0 (c + ctx0*2) → output
        out_in = c + c + ctx_c[0]*2
        self.conv_last = nn.Sequential(
            conv(out_in, c),
            nn.Conv2d(c, 4, 3, 1, 1)  # 3 (RGB residual) + 1 (Mask)
        )

    def forward(self, feats, ctx0=None, ctx1=None):
        """
        Args:
            feats: [f0(c), f1(2c), f2(4c)] from backbone
            ctx0: [wf0(c), wf1(2c), wf2(4c)] warped context of img0 (Phase 2)
            ctx1: [wf0(c), wf1(2c), wf2(4c)] warped context of img1 (Phase 2)
        """
        f0, f1, f2 = feats[0], feats[1], feats[2]
        
        # Concatenate context features if available
        if self.use_context and ctx0 is not None and ctx1 is not None:
            f2 = torch.cat([f2, ctx0[2], ctx1[2]], dim=1)  # (4c + 4c + 4c)
            f1_ctx = torch.cat([ctx0[1], ctx1[1]], dim=1)    # (2c + 2c)
            f0_ctx = torch.cat([ctx0[0], ctx1[0]], dim=1)    # (c + c)
        else:
            f1_ctx = None
            f0_ctx = None
        
        # Upsample F2
        x = self.conv_up0(f2)  # → 2c at H/2
        
        # Cat with F1 (+ context)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        cat_list = [x, f1]
        if f1_ctx is not None:
            cat_list.append(f1_ctx)
        x = torch.cat(cat_list, dim=1)
        
        # Upsample
        x = self.conv_up1(x)  # → c at H
        
        # Cat with F0 (+ context)
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        cat_list = [x, f0]
        if f0_ctx is not None:
            cat_list.append(f0_ctx)
        x = torch.cat(cat_list, dim=1)
        
        # Final prediction
        out = self.conv_last(x)
        
        res = torch.sigmoid(out[:, :3]) * 2 - 1  # Range [-1, 1]
        mask = torch.sigmoid(out[:, 3:4])          # Range [0, 1]
        
        return res, mask