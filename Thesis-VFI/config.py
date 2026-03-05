from functools import partial
import torch.nn as nn

'''==========Model config=========='''
def init_model_config(F=32, W=8, depth=[2, 2, 2], backbone_mode='hybrid', use_ecab=True):
    '''
    Unified Configuration for the NSS-based Hybrid Backbone.
    '''
    return { 
        'embed_dims': [F, 2*F, 4*F],
        'num_heads': [F//16, F//8, F//4], # Scales heads with F
        'mlp_ratios': [4, 4, 4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6), 
        'depths': depth,
        'window_sizes': [W, W, W],
        'backbone_mode': backbone_mode,
        'use_ecab': use_ecab,
        'num_scan_dirs': 4,
        'stripe_width': 4,
        'use_checkpointing': True
    }

# =============================================================================
# Model Variants (NSS + CrossGating)
# =============================================================================

VARIANTS = {
    'base':  init_model_config(F=32, W=8, depth=[2, 2, 2]),
    'hp':    init_model_config(F=48, W=8, depth=[3, 3, 3]),
    'ultra': init_model_config(F=64, W=8, depth=[4, 4, 4])
}

# Update heads for specific variants to match research plan
VARIANTS['hp']['num_heads'] = [3, 6, 12]
VARIANTS['ultra']['num_heads'] = [4, 8, 16]

# =============================================================================
# Phase configurations
# =============================================================================

def get_phase_config(phase, variant='base'):
    cfg = VARIANTS[variant].copy()
    
    if phase == 1:
        logname = f'phase1_nss_{variant}'
        use_flow = False
    elif phase == 2:
        logname = f'phase2_nss_flow_{variant}'
        use_flow = True
    else:
        logname = f'phase3_4k_{variant}'
        use_flow = True
        
    return {
        'LOGNAME': logname,
        'PHASE': phase,
        'MODEL_ARCH': {**cfg, 'use_flow': use_flow},
        'USE_FLOW': use_flow,
        'USE_X4K_TRAINING': (phase == 3)
    }

# Default placeholder
MODEL_CONFIG = get_phase_config(1, 'base')
