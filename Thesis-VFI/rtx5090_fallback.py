"""
Fallback implementation for RTX 5090 sm_120 compatibility
使用纯PyTorch实现SSM，不依赖CUDA kernels

用法：
在 config.py 中设置:
MODEL_ARCH['backbone_mode'] = 'gated_attn_only'  # 纯注意力模式，不使用Mamba2

或者等待 PyTorch/mamba-ssm 更新支持 sm_120
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PureTransformerBackbone(nn.Module):
    """
    纯Transformer实现，完全兼容所有GPU
    性能略低于Mamba2+Hybrid，但可在RTX 5090上运行
    """
    def __init__(self, dim=32, num_layers=4, num_heads=8, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(6, dim, kernel_size=3, stride=1, padding=1)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, window_size)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        """
        x: (B, 6, H, W) - concatenated frame pair
        returns: (B, 3, H, W) - intermediate features
        """
        B, C, H, W = x.shape
        
        # Embed patches
        x = self.patch_embed(x)  # (B, dim, H, W)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Project to output
        x = self.head(x)  # (B, 3, H, W)
        
        return x


class TransformerBlock(nn.Module):
    """单个Transformer block with Gated Window Attention"""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))
        
    def forward(self, x):
        """x: (B, dim, H, W)"""
        B, C, H, W = x.shape
        
        # Reshape for LayerNorm: (B, H, W, C)
        x_perm = x.permute(0, 2, 3, 1)
        
        # Attention branch
        x_perm = x_perm + self.attn(self.norm1(x_perm))
        
        # MLP branch
        x_perm = x_perm + self.mlp(self.norm2(x_perm))
        
        # Back to (B, C, H, W)
        return x_perm.permute(0, 3, 1, 2)


class WindowAttention(nn.Module):
    """Window-based Multi-head Attention"""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (B, H, W, 3*C)
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (3, B, num_heads, H, W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Simplified attention (full attention, not windowed for simplicity)
        q = q.reshape(B * self.num_heads, H * W, C // self.num_heads)
        k = k.reshape(B * self.num_heads, H * W, C // self.num_heads)
        v = v.reshape(B * self.num_heads, H * W, C // self.num_heads)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).reshape(B, self.num_heads, H, W, C // self.num_heads)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        
        out = self.proj(out)
        return out


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# 测试代码
if __name__ == "__main__":
    print("Testing PureTransformerBackbone...")
    
    model = PureTransformerBackbone(dim=32, num_layers=4).cuda()
    x = torch.randn(2, 6, 128, 128).cuda()
    
    with torch.no_grad():
        out = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("\nModel works! Use this as fallback for RTX 5090.")
    print("\nTo enable:")
    print("  1. In config.py, set MODEL_ARCH['backbone_mode'] = 'gated_attn_only'")
    print("  2. Or wait for mamba-ssm sm_120 support")
