import torch
import os
import glob

def check_variant(path):
    try:
        # Load weights only (pkl)
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # Look for patch_embed weight to determine F (Base dimension)
        # Key is likely 'patch_embed.0.weight' or 'module.patch_embed.0.weight'
        f_dim = None
        for key in state_dict.keys():
            if 'patch_embed.0.weight' in key:
                f_dim = state_dict[key].shape[0]
                break
        
        if f_dim == 32:
            return "base (F=32)"
        elif f_dim == 48:
            return "hp (F=48)"
        elif f_dim == 64:
            return "ultra (F=64)"
        else:
            return f"unknown (F={f_dim})"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    ckpt_files = sorted(glob.glob('ckpt/*.pkl'))
    # Filter out _optim.pkl files as they are not model weights
    weights_files = [f for f in ckpt_files if not f.endswith('_optim.pkl')]
    
    print(f"{'Checkpoint File':<40} | {'Detected Variant':<15}")
    print("-" * 60)
    for f in weights_files:
        variant = check_variant(f)
        print(f"{f:<40} | {variant:<15}")
