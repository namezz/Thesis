#!/usr/bin/env python3
"""
环境检查脚本 - 验证训练前的所有依赖和配置
使用方法: python check_environment.py
"""

import os
import sys
from pathlib import Path

def check_mark(passed):
    return "✓" if passed else "✗"

def main():
    print("=" * 60)
    print("  VFI Training Environment Check")
    print("=" * 60)
    print()
    
    all_passed = True
    
    # 1. Check Python version
    print("1. Python Environment")
    py_version = sys.version_info
    py_ok = py_version.major == 3 and py_version.minor >= 8
    print(f"  {check_mark(py_ok)} Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    if not py_ok:
        print("    Required: Python 3.8+")
        all_passed = False
    print()
    
    # 2. Check PyTorch
    print("2. PyTorch")
    try:
        import torch
        torch_ok = True
        print(f"  {check_mark(True)} PyTorch {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"  {check_mark(cuda_available)} CUDA Available: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            print(f"  {check_mark(gpu_count > 0)} GPU Count: {gpu_count}")
            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"    GPU 0: {gpu_name}")
                
                # Check memory
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"    Total Memory: {mem_total:.1f} GB")
                if mem_total < 12:
                    print(f"    ⚠ Warning: Recommended 16GB+ for batch_size=8")
        else:
            print("    ✗ No GPU detected. Training requires CUDA GPU.")
            all_passed = False
    except ImportError as e:
        print(f"  ✗ PyTorch not installed: {e}")
        torch_ok = False
        all_passed = False
    print()
    
    # 3. Check critical dependencies
    print("3. Critical Dependencies")
    deps = [
        ('einops', 'einops'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
    ]
    
    for module, package in deps:
        try:
            __import__(module)
            print(f"  {check_mark(True)} {module}")
        except ImportError:
            print(f"  {check_mark(False)} {module} (install: pip install {package})")
            all_passed = False
    
    # Check Mamba2 (optional but important)
    try:
        from mamba_ssm import Mamba2
        print(f"  {check_mark(True)} mamba_ssm (Mamba2)")
    except ImportError:
        print(f"  {check_mark(False)} mamba_ssm (install: pip install mamba-ssm)")
        print("    ⚠ Critical: Mamba2 is required for the hybrid backbone")
        all_passed = False
    print()
    
    # 4. Check dataset
    print("4. Dataset")
    dataset_paths = [
        "/josh/dataset/vimeo90k/vimeo_triplet",
        "/home/code-server/josh/datasets/video90k/vimeo_septuplet",
    ]
    
    dataset_found = False
    for path in dataset_paths:
        if Path(path).exists():
            seq_path = Path(path) / "sequences"
            if seq_path.exists():
                # Count sequences
                try:
                    num_seqs = len(list(seq_path.glob("*")))
                    print(f"  {check_mark(True)} Dataset found: {path}")
                    print(f"    Sequences: {num_seqs}")
                    dataset_found = True
                    break
                except:
                    pass
    
    if not dataset_found:
        print(f"  {check_mark(False)} Vimeo90K dataset not found")
        print(f"    Checked paths:")
        for path in dataset_paths:
            print(f"      - {path}")
        print(f"    Download from: http://toflow.csail.mit.edu/")
        all_passed = False
    print()
    
    # 5. Check project structure
    print("5. Project Structure")
    required_files = [
        "train.py",
        "Trainer.py",
        "config.py",
        "dataset.py",
        "model/__init__.py",
        "model/backbone.py",
        "model/loss.py",
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        print(f"  {check_mark(exists)} {file}")
        if not exists:
            all_passed = False
    
    # Check directories
    Path("log").mkdir(exist_ok=True)
    Path("ckpt").mkdir(exist_ok=True)
    print(f"  {check_mark(True)} log/ directory")
    print(f"  {check_mark(True)} ckpt/ directory")
    print()
    
    # 6. Check GPU memory (approximate)
    if torch_ok and cuda_available:
        print("6. GPU Memory Test")
        try:
            # Try to allocate a small tensor
            test_tensor = torch.zeros((1, 3, 256, 256), device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            print(f"  {check_mark(True)} GPU memory allocation test passed")
        except Exception as e:
            print(f"  {check_mark(False)} GPU memory allocation failed: {e}")
            all_passed = False
        print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All checks passed! Ready to train.")
        print()
        print("Quick Start Commands:")
        print("  1. Quick test (2 epochs):  bash quick_test.sh")
        print("  2. Full training:          bash start_training_corrected.sh exp1c_hybrid")
        print("  3. Monitor training:       tensorboard --logdir log/")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("  - Install PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("  - Install Mamba: pip install mamba-ssm")
        print("  - Install OpenCV: pip install opencv-python")
        print("  - Download Vimeo90K dataset from http://toflow.csail.mit.edu/")
        return 1

if __name__ == "__main__":
    exit(main())
