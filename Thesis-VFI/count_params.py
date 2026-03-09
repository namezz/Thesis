import torch
import torch.nn as nn
from model import ThesisModel
from config import VARIANTS, get_phase_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def audit_variants():
    results = {}
    for variant_name in ['base', 'hp', 'ultra']:
        # Get config for Phase 1 (No flow)
        config = get_phase_config(1, variant_name)
        
        # Build model
        model = ThesisModel(config['MODEL_ARCH'])
        
        # Count total and backbone only
        total_params = count_parameters(model)
        backbone_params = count_parameters(model.backbone)
        refine_params = count_parameters(model.refine)
        
        results[variant_name] = {
            'total': total_params,
            'backbone': backbone_params,
            'refine': refine_params
        }
        
        # Print results
        print(f"Variant: {variant_name.upper()}")
        print(f"  - F (Base Dim): {config['MODEL_ARCH']['embed_dims'][0]}")
        print(f"  - Depths: {config['MODEL_ARCH']['depths']}")
        print(f"  - Backbone Params: {backbone_params / 1e6:.2f}M")
        print(f"  - RefineNet Params: {refine_params / 1e6:.2f}M")
        print(f"  - Phase 1 Total: {total_params / 1e6:.2f}M")
        print("-" * 30)

if __name__ == "__main__":
    audit_variants()
