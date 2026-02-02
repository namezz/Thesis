from functools import partial
import torch.nn as nn

'''==========Model config=========='''
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4]):
    '''
    Configuration for the Hybrid Backbone.
    Adjust 'depth' and 'F' (channels) to control model size.
    '''
    return { 
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*F//32, 16*F//32],
        'mlp_ratios':[4, 4],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W, W]
    }

MODEL_CONFIG = {
    'LOGNAME': 'hybrid_v1', # Change this for different experiments (e.g., 'hybrid_v1_phase2_flow')
    'MODEL_ARCH': init_model_config(
        F = 32,
        W = 7,
        depth = [2, 2, 2, 4, 4]
    ),
    # Toggle these flags as you progress through phases
    'USE_FLOW': False,        # Phase 2: Set to True
    'USE_X4K_TRAINING': False # Phase 3: Set to True (affects dataloader)
}
