import os
import torch
import argparse
import time
from Trainer import Model
from dataset import VimeoDataset
from torch.utils.data import DataLoader

def dry_run(data_path):
    print("Starting Training Unit Test (Dry Run)...")
    
    # 1. Initialize Model
    # Use local_rank=-1 for non-distributed single GPU test
    model = Model(local_rank=-1)
    model.device()
    
    # 2. Setup Data
    print(f"Loading data from {data_path}...")
    try:
        dataset = VimeoDataset('train', data_path)
        train_data = DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=True, drop_last=True)
        print(f"Dataset loaded. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Training Step Test
    print("Testing training step...")
    for i, imgs in enumerate(train_data):
        if i >= 2: break # Only test 2 steps
        
        imgs = imgs.to(torch.device("cuda"), non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        
        learning_rate = 2e-4
        pred, loss = model.update(imgs, gt, learning_rate, training=True)
        
        print(f"Step {i}: Loss = {loss:.6f}")
        
    # 4. Evaluation Step Test
    print("Testing evaluation step...")
    model.eval()
    for i, imgs in enumerate(train_data):
        if i >= 1: break
        imgs = imgs.to(torch.device("cuda"), non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)
        print(f"Eval Step Done. Pred shape: {pred.shape}")

    # 5. Save Model Test
    print("Testing model saving...")
    if not os.path.exists('ckpt'):
        os.makedirs('ckpt')
    model.save_model(rank=0)
    print("Model saved to ckpt/")

    print("\nUnit Test Completed Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/code-server/josh/datasets/video90k/vimeo_septuplet', help='data path of vimeo90k')
    args = parser.parse_args()
    
    dry_run(args.data_path)
