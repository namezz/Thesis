import torch
import torch.nn as nn
import torch.nn.functional as F


class BackWarp(nn.Module):
    """
    Backward warping as nn.Module.
    Grid is cached per (H, W) — auto-moves with .to(device) and .cuda().
    Supports DDP without global state issues.
    """
    def __init__(self):
        super().__init__()
        self._grid_cache = {}

    def _get_grid(self, H, W, device):
        k = (H, W)
        if k not in self._grid_cache or self._grid_cache[k].device != device:
            hor = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W).expand(1, -1, H, -1)
            ver = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1).expand(1, -1, -1, W)
            self._grid_cache[k] = torch.cat([hor, ver], 1)
        return self._grid_cache[k]

    def forward(self, tenInput, tenFlow):
        H, W = tenFlow.shape[2], tenFlow.shape[3]
        grid = self._get_grid(H, W, tenFlow.device)
        flow_norm = torch.cat([
            tenFlow[:, 0:1] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2] / ((tenInput.shape[2] - 1.0) / 2.0),
        ], 1)
        g = (grid + flow_norm).permute(0, 2, 3, 1)
        return F.grid_sample(tenInput, g, mode='bilinear',
                             padding_mode='border', align_corners=True)


# Backward-compatible functional API
_default_warp = None

def warp(tenInput, tenFlow):
    global _default_warp
    if _default_warp is None:
        _default_warp = BackWarp()
    return _default_warp(tenInput, tenFlow)
