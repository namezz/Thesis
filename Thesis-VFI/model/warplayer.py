import torch
import torch.nn.functional as F

backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    """
    Standard backward warping function.
    Device-agnostic: uses tenFlow's device for grid generation.
    Grid is cached by (device, H, W) — batch-size independent.
    """
    H, W = tenFlow.shape[2], tenFlow.shape[3]
    k = (str(tenFlow.device), H, W)
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, W, device=tenFlow.device).view(
            1, 1, 1, W).expand(1, -1, H, -1)
        tenVertical = torch.linspace(-1.0, 1.0, H, device=tenFlow.device).view(
            1, 1, H, 1).expand(1, -1, -1, W)
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
