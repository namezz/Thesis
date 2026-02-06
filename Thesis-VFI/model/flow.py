import torch
import torch.nn as nn
import torch.nn.functional as F
from .warplayer import warp

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask

class OpticalFlowEstimator(nn.Module):
    """
    IFNet-style 3-scale coarse-to-fine optical flow estimator.
    Reference: RIFE (ECCV 2022), VFIMamba (NeurIPS 2024)
    
    Input: img0, img1 (B, 3, H, W) in [0, 1]
    Output: flow (B, 4, H, W) — bidirectional, timestep-scaled
            mask (B, 1, H, W) — blending logits (pre-sigmoid)
    
    Supports arbitrary timestep for non-midpoint interpolation.
    Supports scale parameter for high-resolution flow estimation.
    """
    def __init__(self):
        super(OpticalFlowEstimator, self).__init__()
        # block0: 6 (img0+img1) + 1 (timestep map) = 7 input channels
        self.block0 = IFBlock(7, c=240)
        # block1/2: 13 (img0+img1+warped0+warped1+mask) + 4 (flow) + 1 (timestep) = 18
        self.block1 = IFBlock(18, c=150)
        self.block2 = IFBlock(18, c=90)

    def forward(self, img0, img1, timestep=0.5, scale_list=[4, 2, 1]):
        """
        Args:
            img0, img1: (B, 3, H, W) in [0, 1]
            timestep: float or (B, 1, 1, 1) tensor, default 0.5
            scale_list: multi-scale factors, e.g. [4,2,1] or [8,4,2] for high-res
        Returns:
            flow: (B, 4, H, W) — bidirectional flow scaled by timestep
            mask: (B, 1, H, W) — blending logits (pre-sigmoid)
        """
        B = img0.shape[0]
        # Build timestep map (B, 1, H, W)
        if isinstance(timestep, (int, float)):
            timestep_map = img0.new_full((B, 1, img0.shape[2], img0.shape[3]), timestep)
        else:
            timestep_map = timestep
        
        # Scale 0 (coarsest): initial flow from raw images + timestep
        x = torch.cat((img0, img1, timestep_map), 1)
        flow, mask = self.block0(x, None, scale=scale_list[0])
        
        # Scale 1: refine with warped images
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        x = torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep_map), 1)
        flow_d, mask_d = self.block1(x, flow, scale=scale_list[1])
        flow = flow + flow_d
        mask = mask + mask_d
        
        # Scale 2 (finest): final refinement
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        x = torch.cat((img0, img1, warped_img0, warped_img1, mask, timestep_map), 1)
        flow_d, mask_d = self.block2(x, flow, scale=scale_list[2])
        flow = flow + flow_d
        mask = mask + mask_d
        
        return flow, mask

def build_flow_estimator():
    return OpticalFlowEstimator()
