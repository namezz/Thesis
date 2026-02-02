import torch
import torch.nn as nn

# Refinement Network (Feature -> Image)

class RefineNet(nn.Module):
    def __init__(self, c=32):
        super(RefineNet, self).__init__()
        self.conv_out = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, feat):
        # TODO: Implement UNet-like refinement
        return torch.sigmoid(self.conv_out(feat))
