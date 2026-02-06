import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

# ============================================================
# Laplacian Pyramid Loss (RIFE / VFIMamba / EMA-VFI standard)
# ============================================================

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

# ============================================================
# VGG Perceptual Loss (multi-layer feature matching)
# ============================================================

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

# ============================================================
# Ternary (Census) Loss — device-agnostic version
# Used by: RIFE, EMA-VFI, IFRNet (proven effective for VFI)
# Captures local structural patterns robust to illumination changes
# ============================================================

class Ternary(nn.Module):
    def __init__(self, patch_size=7):
        super(Ternary, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        w = np.transpose(w, (3, 2, 0, 1))
        self.register_buffer('w', torch.tensor(w).float())

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        w = self.w.to(dtype=tensor_.dtype)
        patches = F.conv2d(tensor_, w, padding=self.patch_size // 2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss

# ============================================================
# Charbonnier Loss (robust L1, used by IFRNet)
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps))

# ============================================================
# Flow Smoothness Loss — encourages spatial smoothness in flow
# Edge-aware: weighted by image gradient to allow sharp flow at edges
# ============================================================

class FlowSmoothnessLoss(nn.Module):
    """Edge-aware flow smoothness regularization for Phase 2+."""
    def forward(self, flow, img):
        # Image gradient for edge-aware weighting
        img_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        img_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        weight_x = torch.exp(-img_dx)
        weight_y = torch.exp(-img_dy)
        
        # Flow gradient
        flow_dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        flow_dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        
        loss = (weight_x * flow_dx).mean() + (weight_y * flow_dy).mean()
        return loss

# ============================================================
# Composite Loss — phase-aware loss combination
# ============================================================

class CompositeLoss(nn.Module):
    """
    Phase-aware composite loss for VFI training.
    
    Phase 1: LapLoss + Ternary + VGG (backbone-only, no flow)
    Phase 2: LapLoss + Ternary + VGG + FlowSmoothness (flow guidance)
    Phase 3: Same as Phase 2 (4K fine-tune)
    
    Loss weights from reference implementations:
    - LapLoss: 1.0 (primary, RIFE/VFIMamba/EMA-VFI standard)
    - Ternary: 1.0 (proven in RIFE/IFRNet, captures structural similarity)
    - VGG:    0.005 (perceptual quality, conservative weight to avoid hallucination)
    - FlowSmooth: 0.1 (Phase 2+, edge-aware regularization)
    """
    def __init__(self, phase=1, w_lap=1.0, w_ter=1.0, w_vgg=0.005, w_flow_smooth=0.1):
        super(CompositeLoss, self).__init__()
        self.phase = phase
        self.w_lap = w_lap
        self.w_ter = w_ter
        self.w_vgg = w_vgg
        self.w_flow_smooth = w_flow_smooth
        
        self.lap_loss = LapLoss()
        self.ternary_loss = Ternary()
        self.vgg_loss = VGGPerceptualLoss()
        if phase >= 2:
            self.flow_smooth_loss = FlowSmoothnessLoss()
    
    def forward(self, pred, gt, flow=None, img0=None):
        """
        Args:
            pred: predicted frame (B, 3, H, W)
            gt: ground truth frame (B, 3, H, W)
            flow: optical flow (B, 4, H, W) — only for Phase 2+
            img0: input frame 0 (B, 3, H, W) — for edge-aware flow smoothness
        Returns:
            total_loss, loss_dict (for TensorBoard logging)
        """
        loss_lap = self.lap_loss(pred, gt)
        loss_ter = self.ternary_loss(pred, gt)
        loss_vgg = self.vgg_loss(pred, gt)
        
        total = self.w_lap * loss_lap + self.w_ter * loss_ter + self.w_vgg * loss_vgg
        
        loss_dict = {
            'loss_lap': loss_lap.item(),
            'loss_ter': loss_ter.item(),
            'loss_vgg': loss_vgg.item(),
        }
        
        if self.phase >= 2 and flow is not None and img0 is not None:
            loss_flow_smooth = self.flow_smooth_loss(flow, img0)
            total = total + self.w_flow_smooth * loss_flow_smooth
            loss_dict['loss_flow_smooth'] = loss_flow_smooth.item()
        
        loss_dict['loss_total'] = total.item()
        return total, loss_dict
