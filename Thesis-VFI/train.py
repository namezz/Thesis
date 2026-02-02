import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from Trainer import Model
from dataset import VimeoDataset, X4KDataset, MixedDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from config import *

device = torch.device("cuda")

def get_learning_rate(step, step_per_epoch):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (2e-4 - 2e-5) * mul + 2e-5

from tqdm import tqdm

def train(model, local_rank, batch_size, data_path, x4k_path=None, mixed_ratio=(2, 1)):
    if local_rank == 0:
        writer = SummaryWriter(f'log/train_{MODEL_CONFIG["LOGNAME"]}')
    step = 0
    nr_eval = 0
    
    vimeo_train = VimeoDataset('train', data_path)
    if MODEL_CONFIG.get('USE_X4K_TRAINING', False) and x4k_path:
        if local_rank == 0: print(f"Phase 3: Enabling Mixed Training (Vimeo + X4K) with ratio {mixed_ratio}")
        x4k_train = X4KDataset('train', x4k_path)
        dataset = MixedDataset(vimeo_train, x4k_train, ratio=mixed_ratio)
    else:
        dataset = vimeo_train
        
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True, sampler=sampler)
    step_per_epoch = train_data.__len__()
    
    dataset_val = VimeoDataset('test', data_path)
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=4)
    
    if local_rank == 0: print(f'Training with {MODEL_CONFIG["LOGNAME"]}...')
    time_stamp = time.time()
    for epoch in range(300):
        sampler.set_epoch(epoch)
        # Use tqdm for progress bar
        pbar = tqdm(train_data, desc=f"Epoch {epoch}", disable=(local_rank != 0))
        for i, imgs in enumerate(pbar):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step, step_per_epoch)
            _, loss = model.update(imgs, gt, learning_rate, training=True)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss', loss, step)
            
            # Update tqdm postfix instead of print
            if local_rank == 0:
                pbar.set_postfix({'loss': f'{loss:.4e}', 'lr': f'{learning_rate:.2e}'})
                
            step += 1
        nr_eval += 1
        if nr_eval % 3 == 0:
            evaluate(model, val_data, nr_eval, local_rank)
        model.save_model(local_rank)    
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank):
    if local_rank == 0:
        writer_val = SummaryWriter(f'log/validate_{MODEL_CONFIG["LOGNAME"]}')

    psnr = []
    for _, imgs in enumerate(val_data):
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
   
    psnr = np.array(psnr).mean()
    if local_rank == 0:
        print(f"Evaluation Epoch {nr_eval}, PSNR: {psnr:.4f}")
        writer_val.add_scalar('psnr', psnr, nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--local-rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_path', type=str, required=True, help='data path of vimeo90k')
    parser.add_argument('--x4k_path', type=str, default=None, help='data path of x4k1000fps')
    parser.add_argument('--mixed_ratio', type=str, default="2:1", help='Mixed ratio Vimeo:X4K (e.g. 2:1)')
    args = parser.parse_args()
    
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = args.local_rank
    
    v_w, x_w = map(int, args.mixed_ratio.split(':'))
    
    dist.init_process_group(backend="nccl", world_size=args.world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    
    if local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
        
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    model = Model(local_rank)
    train(model, local_rank, args.batch_size, args.data_path, args.x4k_path, mixed_ratio=(v_w, x_w))