from functools import partial
import torch.nn as nn

'''==========Model config=========='''
def init_model_config(F=32, W=8, depth=[2, 2, 2], backbone_mode='hybrid', use_mhc=False, use_ecab=True):
    '''
    Configuration for the Hybrid Backbone.
    
    Args:
        F: Base feature dimension
        W: Window size for attention
        depth: Number of blocks per stage
        backbone_mode: 'hybrid' | 'mamba2_only' | 'gated_attn_only'
        use_mhc: Whether to use Manifold Hyper-Connections
        use_ecab: Whether to use ECAB (vs standard CAB)
    '''
    return { 
        'embed_dims': [F, 2*F, 4*F],  # 3 scales
        'num_heads': [2, 4, 8],        # Heads per scale
        'mlp_ratios': [4, 4, 4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6), 
        'depths': depth,
        'window_sizes': [W, W, W],
        # Ablation control
        'backbone_mode': backbone_mode,  # 'hybrid', 'mamba2_only', 'gated_attn_only'
        'use_mhc': use_mhc,
        'use_ecab': use_ecab,
    }

# =============================================================================
# Phase-specific Configurations
# =============================================================================

PHASE1_CONFIG = {
    'LOGNAME': 'phase1_hybrid',
    'PHASE': 1,
    'MODEL_ARCH': {**init_model_config(F=32, W=8, depth=[2, 2, 2], backbone_mode='hybrid'), 'use_flow': False},
    'USE_FLOW': False,
    'USE_X4K_TRAINING': False,
}

PHASE2_CONFIG = {
    'LOGNAME': 'phase2_flow',
    'PHASE': 2,
    'MODEL_ARCH': {**init_model_config(F=32, W=8, depth=[2, 2, 2], backbone_mode='hybrid'), 'use_flow': True},
    'USE_FLOW': True,
    'USE_X4K_TRAINING': False,
}

PHASE3_CONFIG = {
    'LOGNAME': 'phase3_4k',
    'PHASE': 3,
    'MODEL_ARCH': {**init_model_config(F=32, W=8, depth=[2, 2, 2], backbone_mode='hybrid'), 'use_flow': True},
    'USE_FLOW': True,
    'USE_X4K_TRAINING': True,
}

# =============================================================================
# Ablation Experiment Configurations (Phase 1)
# =============================================================================

ABLATION_CONFIGS = {
    # Exp-1a: Pure Mamba2 (no attention)
    'exp1a_mamba2_only': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='mamba2_only'),
    # Exp-1b: Pure Gated Window Attention (no SSM)
    'exp1b_gated_attn_only': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='gated_attn_only'),
    # Exp-1c: Hybrid LGS Block (main)
    'exp1c_hybrid': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='hybrid'),
    # Exp-1f: ECAB vs CAB
    'exp1f_ecab': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='hybrid', use_ecab=True),
    'exp1f_cab': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='hybrid', use_ecab=False),
    # Exp-1g: Gated vs Non-gated (handled in GatedWindowAttention)
    'exp1g_gated': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='hybrid'),
    # Exp-1h: mHC vs Standard Residual
    'exp1h_mhc': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='hybrid', use_mhc=True),
    'exp1h_standard': init_model_config(F=32, W=8, depth=[2,2,2], backbone_mode='hybrid', use_mhc=False),
    # Exp-1i: Window size ablation
    'exp1i_win7': init_model_config(F=32, W=7, depth=[2,2,2], backbone_mode='hybrid'),
    'exp1i_win14': init_model_config(F=32, W=14, depth=[2,2,2], backbone_mode='hybrid'),
}

# Default active config (can be overridden by train.py args)
MODEL_CONFIG = {
    'LOGNAME': 'hybrid_v1_baseline',
    'MODEL_ARCH': {**init_model_config(
        F = 32,
        W = 8,
        depth = [2, 2, 2]
    ), 'use_flow': False},
    'USE_FLOW': False,
    'USE_X4K_TRAINING': False
}
