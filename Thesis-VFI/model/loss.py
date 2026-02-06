import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

def gauss_kernel(channels=3, device=None):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if device is not None:
        kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros_like(x)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        if up.shape != current.shape:
            up = F.interpolate(up, size=current.shape[-2:], mode='bilinear', align_corners=False)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import models
        from torchvision.models import VGG19_Weights
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(9)]) # relu2_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(9, 18)]) # relu3_3
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(18, 27)]) # relu4_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        if input.device != self.mean.device:
            self.mean = self.mean.to(input.device)
            self.std = self.std.to(input.device)
            self.slice1 = self.slice1.to(input.device)
            self.slice2 = self.slice2.to(input.device)
            self.slice3 = self.slice3.to(input.device)
            
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        h1 = self.slice1(input)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        
        with torch.no_grad():
            h1_target = self.slice1(target)
            h2_target = self.slice2(h1_target)
            h3_target = self.slice3(h2_target)
            
        loss = F.l1_loss(h1, h1_target) + F.l1_loss(h2, h2_target) + F.l1_loss(h3, h3_target)
        return loss

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.channels = channels
        
    def forward(self, input, target):
        kernel = gauss_kernel(channels=self.channels, device=input.device)
        pyr_input  = laplacian_pyramid(img=input, kernel=kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

class Ternary(nn.Module):
    def __init__(self, device):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant, more robust than L1 near zero)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))
