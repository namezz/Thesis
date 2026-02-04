import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import to_2tuple, trunc_normal_

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ECAB(nn.Module):
    """
    Efficient Channel Attention Block (ECA-Net, CVPR 2020)
    Replaces standard CAB (SE-Block) for better efficiency and performance.
    Avoids dimensionality reduction and uses 1D convolution for cross-channel interaction.
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECAB, self).__init__()
        # Adaptive kernel size k
        k = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        self.padding = kernel_size // 2
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=self.padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W) or (B, L, C) -> needs handling
        is_sequence = False
        if x.dim() == 3: # (B, L, C)
            is_sequence = True
            B, L, C = x.shape
            # Reshape to (B, C, L, 1) or similar for pooling, 
            # but since we want Global Average Pooling over L, we can just mean(1)
            y = x.mean(1).view(B, 1, C) # (B, 1, C)
        else:
            y = self.avg_pool(x) # (B, C, 1, 1)
            y = y.squeeze(-1).permute(0, 2, 1) # (B, 1, C)

        y = self.conv(y) # (B, 1, C)
        y = self.sigmoid(y) # (B, 1, C)
        y = y.permute(0, 2, 1) # (B, C, 1)
        
        if is_sequence:
            return x * y.transpose(1, 2) # (B, L, C) * (B, 1, C) -> broadcast?
            # y is (B, C, 1). Transpose to (B, 1, C) to match (B, L, C)
            # wait, y is (B, C, 1). x is (B, L, C). 
            # We want to scale C.
            # y.view(B, 1, C) -> (B, L, C) broadcast OK.
            return x * y.view(B, 1, C)
        else:
            return x * y.unsqueeze(-1) # (B, C, H, W) * (B, C, 1, 1)

def scan_images(x):
    """
    SS2D Scan: 4 directions (Original, Flip H, Flip V, Flip HV)
    Input: (B, H, W, C)
    Output: (4*B, L, C) where L = H*W
    """
    B, H, W, C = x.shape
    
    # 1. Original
    x0 = x.flatten(1, 2) # (B, L, C)
    
    # 2. Flip H
    x1 = x.flip(2).flatten(1, 2)
    
    # 3. Flip V
    x2 = x.flip(1).flatten(1, 2)
    
    # 4. Flip HV (Rotate 180)
    x3 = x.flip(1, 2).flatten(1, 2)
    
    # Batch them together for Mamba efficiency
    x_scan = torch.cat([x0, x1, x2, x3], dim=0) # (4B, L, C)
    return x_scan

def merge_images(x_scan, B, H, W):

    """

    SS2D Merge: Reverse scanning and sum/average

    Input: (4*B, L, C)

    Output: (B, H, W, C)

    """

    C = x_scan.shape[-1]

    

    # Split back to 4 branches

    x0, x1, x2, x3 = x_scan.chunk(4, dim=0) # Each (B, L, C)

    

    # Reshape and un-flip

    x0 = x0.view(B, H, W, C)

    x1 = x1.view(B, H, W, C).flip(2)

    x2 = x2.view(B, H, W, C).flip(1)

    x3 = x3.view(B, H, W, C).flip(1, 2)

    

    # Sum (or Average)

    # Standard practice is sum, then linear projection handles scaling

    return x0 + x1 + x2 + x3



def sinkhorn_knopp(W, iterations=20):

    """

    Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff Polytope 

    (doubly stochastic matrices).

    Input W: (..., M, N)

    """

    for _ in range(iterations):

        # Normalize rows

        W = W / (W.sum(dim=-1, keepdim=True) + 1e-6)

        # Normalize columns

        W = W / (W.sum(dim=-2, keepdim=True) + 1e-6)

    return W



class ManifoldResConnection(nn.Module):

    """

    mHC: Manifold-Constrained Hyper-Connections (DeepSeek, 2025)

    Constrains residual mixing matrix to the Birkhoff Polytope using Sinkhorn-Knopp.

    """

    def __init__(self, dim, num_streams=2):

        super().__init__()

        self.dim = dim

        self.num_streams = num_streams

        # Mixing matrix: (num_streams, num_streams)

        # Initialize as identity or close to it

        self.mixing_weights = nn.Parameter(torch.eye(num_streams))

        

    def forward(self, streams):

        """

        streams: List of [B, L, C] tensors

        """

        # 1. Project mixing weights to Birkhoff Polytope

        # Ensure positive weights before Sinkhorn

        W = torch.exp(self.mixing_weights) 

        W_stochastic = sinkhorn_knopp(W)

        

        # 2. Mix streams

        # streams_tensor: (num_streams, B, L, C)

        streams_tensor = torch.stack(streams, dim=0)

        

        # result = sum_i(W_i * stream_i) logic but for multiple output streams if needed.

        # Here we usually just want one output residual stream.

        # If we follow mHC's hyper-connection logic, we might maintain multiple streams.

        # For simplicity in VFI backbone, we use it to mix Branch A and Branch B outputs.

        

        # out = \sum_j W_{0,j} * stream_j

        out = 0

        for j in range(self.num_streams):

            out = out + W_stochastic[0, j] * streams[j]

            

        return out



def matvlm_init_mamba2(mamba_layer, attn_layer):

    """

    MaTVLM initialization strategy (ICCV 2025):

    Initialize Mamba2 (x, B, C) linear weights from Attention (V, K, Q).

    """

    with torch.no_grad():

        # Mamba2 in_proj: (dim, 2 * d_inner + 2 * d_state + dt_rank) roughly

        # This is implementation dependent. Official mamba_ssm.Mamba2 has a large in_proj.

        # If using simplified Mamba2, we map:

        # V -> x (input/value)

        # K -> B (key/state)

        # Q -> C (query/output)

        

        # Note: This requires internal access to weights which might differ 

        # based on mamba_ssm version.

        try:

            # Placeholder for exact mapping logic

            # dim = mamba_layer.d_model

            # mamba_layer.in_proj.weight[:dim] = attn_layer.qkv.weight[2*dim:] # V -> x

            pass

        except Exception as e:

            print(f"MaTVLM init failed: {e}")