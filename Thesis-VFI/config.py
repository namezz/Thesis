from functools import partial
import torch.nn as nn

'''==========Model config=========='''
def init_model_config(F=32, W=8, depth=[2, 2, 2]):
    '''
    Configuration for the Hybrid Backbone (Phase 1).
    '''
    return { 
        'embed_dims':[F, 2*F, 4*F], # 3 scales
        'num_heads':[2, 4, 8],      # Heads per scale
        'mlp_ratios':[4, 4, 4],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W, W, W]
    }

MODEL_CONFIG = {
    'LOGNAME': 'hybrid_v1_baseline',
    'MODEL_ARCH': init_model_config(
        F = 32,
        W = 8,
        depth = [2, 2, 2] # Lightweight baseline
    ),
    'USE_FLOW': False,
    'USE_X4K_TRAINING': False
}
