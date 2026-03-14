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
import config as project_config
from benchmark.utils.pytorch_msssim import ssim_matlab

device = torch.device("cuda")

def get_learning_rate(step, step_per_epoch, total_epochs=300, base_lr=2e-4, min_lr=2e-5):
    if step < 2000:
        mul = step / 2000
        return base_lr * mul
    else:
        mul = np.cos((step - 2000) / (total_epochs * step_per_epoch - 2000) * math.pi) * 0.5 + 0.5
        return (base_lr - min_lr) * mul + min_lr

from tqdm import tqdm

def train(model, local_rank, batch_size, data_path, x4k_path=None, mixed_ratio=(2, 1), freeze_epochs=0, total_epochs=300, curriculum=False, curriculum_T=50, crop_size=256, lr=2e-4, num_workers=8, grad_accum=1, eval_interval=10, train_state=None):
    logname = project_config.MODEL_CONFIG["LOGNAME"]
    if local_rank == 0:
        writer = SummaryWriter(f'log/train_{logname}')
        writer_val = SummaryWriter(f'log/validate_{logname}')
    else:
        writer = writer_val = None
    
    # Initialize state from train_state or scratch
    step = 0
    nr_eval = 0
    best_psnr_holder = {'val': 0.0}
    start_epoch = 0
    
    if train_state is not None:
        start_epoch = train_state.get('epoch', 0) + 1
        step = train_state.get('step', 0)
        nr_eval = train_state.get('nr_eval', 0)
        best_psnr_holder['val'] = train_state.get('best_psnr', 0.0)
        if local_rank == 0:
            print(f"★ Resuming from epoch {start_epoch}, step {step}, best_psnr {best_psnr_holder['val']:.4f}")
    
    # Freeze backbone if requested
    if freeze_epochs > 0:
        model.freeze_backbone()
    
    vimeo_train = VimeoDataset('train', data_path)
    vimeo_train.h = vimeo_train.w = crop_size
    if project_config.MODEL_CONFIG.get('USE_X4K_TRAINING', False) and x4k_path:
        if local_rank == 0: print(f"Phase 3: Mixed Training enabled with ratio {mixed_ratio}")
        x4k_train = X4KDataset('train', x4k_path)
        dataset = MixedDataset(vimeo_train, x4k_train, ratio=mixed_ratio)
    else:
        dataset = vimeo_train
    
    if curriculum:
        curriculum_sizes = [256, 384, 512]
        if local_rank == 0:
            print(f"Curriculum learning enabled: sizes={curriculum_sizes}")
        
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=sampler)
    step_per_epoch = train_data.__len__()
    
    dataset_val = VimeoDataset('test', data_path)
    
    # 新增：讓多卡平分驗證集 (單卡時會自動全吃)
    val_sampler = DistributedSampler(dataset_val, shuffle=False) 
    
    # 修改：將 batch_size 降為 1，並放入 sampler
    val_data = DataLoader(dataset_val, batch_size=1, pin_memory=True, num_workers=num_workers, sampler=val_sampler)
    
    if local_rank == 0:
        print(f'Training with {logname}...')
        if grad_accum > 1:
            print(f'Gradient accumulation: {grad_accum} steps, effective batch = {batch_size * grad_accum}')
            
    time_stamp = time.time()
    for epoch in range(start_epoch, total_epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            model.unfreeze_backbone()
            if local_rank == 0: print(f"Epoch {epoch}: Backbone unfrozen")
        
        sampler.set_epoch(epoch)
        torch.cuda.empty_cache()
        model.loss_fn.current_epoch = epoch
        
        if curriculum:
            if epoch < curriculum_T: crop_idx = 0
            elif epoch < curriculum_T * 2: crop_idx = 1
            else: crop_idx = 2
            new_crop = curriculum_sizes[min(crop_idx, 2)]
            if hasattr(dataset, 'vimeo'):
                dataset.vimeo.h = dataset.vimeo.w = new_crop
                if hasattr(dataset, 'x4k'): dataset.x4k.h = dataset.x4k.w = new_crop
            else:
                dataset.h = dataset.w = new_crop
            
            if hasattr(dataset, 'set_x4k_prob'):
                x4k_prob = 0.5 / (1.0 + math.exp(-0.2 * (epoch - curriculum_T)))
                dataset.set_x4k_prob(x4k_prob)
        
        pbar = tqdm(train_data, desc=f"Epoch {epoch}", disable=(local_rank != 0))
        for i, imgs in enumerate(pbar):
            if args.dry_run and i >= 10: break # Exit early for dry run
            imgs = imgs.to(device, non_blocking=True) / 255.
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]
            learning_rate = get_learning_rate(step, step_per_epoch, total_epochs, base_lr=lr)
            
            if grad_accum > 1:
                _, loss_dict = model.update(imgs, gt, learning_rate, training=True, accumulate=True, grad_accum=grad_accum)
                if (i + 1) % grad_accum == 0 or (i + 1) == len(train_data):
                    model.accum_step()
            else:
                _, loss_dict = model.update(imgs, gt, learning_rate, training=True, accumulate=False)
            
            if step % 200 == 1 and local_rank == 0 and writer is not None:
                writer.add_scalar('learning_rate', learning_rate, step)
                for k, v in loss_dict.items(): writer.add_scalar(f'loss/{k}', v, step)
            
            if local_rank == 0:
                pbar.set_postfix({'loss': f'{loss_dict.get("loss_total", 0):.4e}', 'lr': f'{learning_rate:.2e}'})
            
            # Explicitly delete large tensors to free memory for the next step
            del imgs, gt
            
            step += 1
            
        # 迴圈剛結束的地方
        if local_rank == 0:
            print(f"========== DEBUG: 準備進入 Evaluation (Epoch {epoch}) ==========", flush=True)
            
        nr_eval += 1
        if nr_eval % eval_interval == 0:
            evaluate(model, val_data, nr_eval, local_rank, writer_val=writer_val, best_psnr_holder=best_psnr_holder, dry_run=args.dry_run)
        
        if local_rank == 0:
            print(f"========== DEBUG: Evaluation 完成，準備儲存 Checkpoint ==========", flush=True)
        
        train_state = {'epoch': epoch, 'step': step, 'nr_eval': nr_eval, 'best_psnr': best_psnr_holder['val']}
        model.save_model(local_rank, train_state=train_state)    
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val=None, best_psnr_holder=None, dry_run=False):
    psnr_list, ssim_list = [], []
    for i, imgs in enumerate(val_data):
        if dry_run and i >= 2: break # Only 2 samples for dry run
        imgs = imgs.to(device, non_blocking=True) / 255.
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]
        with torch.no_grad():
            pred, _ = model.update(imgs, gt, training=False)
        for j in range(gt.shape[0]):
            psnr_list.append(-10 * math.log10(((gt[j] - pred[j])**2).mean().cpu().item()))
            ssim_list.append(ssim_matlab(gt[j:j+1], pred[j:j+1]).cpu().item())
        del imgs, gt, pred
   
    avg_psnr, avg_ssim = np.mean(psnr_list), np.mean(ssim_list)
    if local_rank == 0:
        print(f"Evaluation Epoch {nr_eval}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        if writer_val:
            writer_val.add_scalar('psnr', avg_psnr, nr_eval)
            writer_val.add_scalar('ssim', avg_ssim, nr_eval)
        if avg_psnr > best_psnr_holder['val']:
            best_psnr_holder['val'] = avg_psnr
            model.save_model(rank=0, suffix='_best')
            print(f"  ★ New best PSNR: {avg_psnr:.4f}")
    torch.cuda.empty_cache()
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--x4k_path', type=str, default=None)
    parser.add_argument('--mixed_ratio', type=str, default="2:1")
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=3)
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--variant', type=str, default='base', choices=['base', 'hp', 'ultra'])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--freeze_backbone', type=int, default=0)
    parser.add_argument('--backbone_lr_scale', type=float, default=1.0)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--curriculum_T', type=int, default=50)
    args = parser.parse_args()
    
    # Enhanced Distributed Initialization
    if 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(0)
        dist.init_process_group(backend="nccl", init_method='tcp://127.0.0.1:29505', world_size=1, rank=0)

    v_w, x_w = map(int, args.mixed_ratio.split(':'))
    
    # Unified Config Generation
    project_config.MODEL_CONFIG = project_config.get_phase_config(args.phase, args.variant)
    if args.exp_name:
        project_config.MODEL_CONFIG['LOGNAME'] = args.exp_name
    
    if local_rank == 0:
        print("="*40)
        print(f"DEBUG: Selected Variant: {args.variant}")
        print(f"DEBUG: Final LogName: {project_config.MODEL_CONFIG['LOGNAME']}")
        print(f"DEBUG: Model F dim: {project_config.MODEL_CONFIG['MODEL_ARCH']['embed_dims'][0]}")
        print("="*40)
    
    seed = 1234
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    # Pass the actual generated config to the trainer
    model = Model(local_rank, project_config.MODEL_CONFIG, backbone_lr_scale=args.backbone_lr_scale)
    
    # Auto-resume logic
    train_state = None
    if os.path.exists(f'ckpt/{project_config.MODEL_CONFIG["LOGNAME"]}.pkl'):
        train_state = model.load_model(name=project_config.MODEL_CONFIG["LOGNAME"])
    elif args.resume:
        train_state = model.load_model(name=args.resume.replace('.pkl', ''))
    
    train(model, local_rank, args.batch_size, args.data_path, args.x4k_path, 
          mixed_ratio=(v_w, x_w), freeze_epochs=args.freeze_backbone, 
          total_epochs=(1 if args.dry_run else args.epochs),
          curriculum=args.curriculum, curriculum_T=args.curriculum_T,
          crop_size=args.crop_size, lr=args.lr, num_workers=args.num_workers,
          grad_accum=args.grad_accum, eval_interval=args.eval_interval,
          train_state=train_state)
