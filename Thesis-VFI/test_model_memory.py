#!/usr/bin/env python
"""
Memory test for Thesis VFI Model
Tests different batch sizes and input resolutions to find optimal settings
"""
import torch
import torch.nn as nn
from model import ThesisModel
from config import PHASE1_CONFIG
import gc

def test_memory(batch_size, height, width):
    """Test model with given batch size and resolution"""
    print(f"\n{'='*60}")
    print(f"Testing: batch_size={batch_size}, resolution={height}x{width}")
    print(f"{'='*60}")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Create model
        cfg = PHASE1_CONFIG['MODEL_ARCH'].copy()
        model = ThesisModel(cfg).cuda()
        model.train()
        
        # Create dummy input (6 channels: 2 frames × 3 RGB)
        x = torch.randn(batch_size, 6, height, width).cuda()
        gt = torch.randn(batch_size, 3, height, width).cuda()
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.cuda.amp.autocast():
            pred = model(x)
        
        mem_forward = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✓ Forward pass successful")
        print(f"  Peak memory after forward: {mem_forward:.2f} GB")
        
        # Test backward pass
        print("Testing backward pass...")
        loss = nn.L1Loss()(pred, gt)
        loss.backward()
        
        mem_backward = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✓ Backward pass successful")
        print(f"  Peak memory after backward: {mem_backward:.2f} GB")
        
        # Clear
        del model, x, gt, pred, loss
        torch.cuda.empty_cache()
        gc.collect()
        
        return True, mem_backward
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"✗ OOM Error: {str(e)[:100]}...")
            # Clear
            torch.cuda.empty_cache()
            gc.collect()
            return False, 0
        else:
            raise e

if __name__ == "__main__":
    print("=" * 60)
    print("  VFI Model Memory Test")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()
    
    # Test configurations (batch_size, height, width)
    configs = [
        (1, 256, 256),   # Minimal
        (2, 256, 256),   # Dry run default
        (4, 256, 256),   # Original dry run
        (1, 128, 128),   # Very small
        (2, 128, 128),   # Small
        (4, 128, 128),   # Small batch
    ]
    
    successful = []
    failed = []
    
    for bs, h, w in configs:
        success, mem = test_memory(bs, h, w)
        if success:
            successful.append((bs, h, w, mem))
        else:
            failed.append((bs, h, w))
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n✓ Successful configurations ({len(successful)}):")
    for bs, h, w, mem in successful:
        print(f"  - batch_size={bs}, {h}x{w}: {mem:.2f} GB peak memory")
    
    if failed:
        print(f"\n✗ Failed configurations ({len(failed)}):")
        for bs, h, w in failed:
            print(f"  - batch_size={bs}, {h}x{w}")
    
    # Recommendation
    if successful:
        print("\n" + "=" * 60)
        print("  RECOMMENDATION")
        print("=" * 60)
        # Find largest successful batch size at 256x256
        best_256 = [cfg for cfg in successful if cfg[1] == 256 and cfg[2] == 256]
        if best_256:
            bs, h, w, mem = max(best_256, key=lambda x: x[0])
            print(f"For training at 256x256: use batch_size={bs}")
            print(f"  (Peak memory: {mem:.2f} GB, leaves {16 - mem:.2f} GB headroom)")
        else:
            # Try smaller resolution
            best_128 = [cfg for cfg in successful if cfg[1] == 128 and cfg[2] == 128]
            if best_128:
                bs, h, w, mem = max(best_128, key=lambda x: x[0])
                print(f"WARNING: 256x256 failed. Try training at 128x128 first:")
                print(f"  Use batch_size={bs} at {h}x{w}")
                print(f"  (Peak memory: {mem:.2f} GB)")
