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
        if flow != None:
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
    def __init__(self):
        super(OpticalFlowEstimator, self).__init__()
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(17, c=150)
        self.block2 = IFBlock(17, c=90)

    def forward(self, img0, img1, timestep=0.5):
        # timestep placeholder for now
        x = torch.cat((img0, img1), 1)
        flow0, mask0 = self.block0(x, None, scale=4)
        
        warped_img0 = warp(img0, flow0[:, :2])
        warped_img1 = warp(img1, flow0[:, 2:4])
        x = torch.cat((img0, img1, warped_img0, warped_img1, mask0), 1)
        flow1, mask1 = self.block1(x, flow0, scale=2)
        flow1 = flow0 + flow1
        mask1 = mask0 + mask1
        
        warped_img0 = warp(img0, flow1[:, :2])
        warped_img1 = warp(img1, flow1[:, 2:4])
        x = torch.cat((img0, img1, warped_img0, warped_img1, mask1), 1)
        flow2, mask2 = self.block2(x, flow1, scale=1)
        flow2 = flow1 + flow2
        mask2 = mask1 + mask2
        
        return flow2, mask2

def build_flow_estimator():
    return OpticalFlowEstimator()
