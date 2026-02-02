import cv2
import math
import sys
import torch
import numpy as np
import argparse
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'


'''==========Model setting=========='''
TTA = True
if args.model == 'ours_small':
    TTA = False
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 16,
        depth = [2, 2, 2, 2, 2]
    )
else:
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
model = Model(-1)
model.load_model()
model.eval()
model.device()


print(f'=========================Starting testing=========================')
print(f'Dataset: Vimeo90K   Model: {model.name}   TTA: {TTA}')
path = args.path
f = open(path + '/tri_testlist.txt', 'r')
psnr_list, ssim_list = [], []
for i in f:
    name = str(i).strip()
    if(len(name) <= 1):
        continue
    
    # 建議先印出路徑，確認路徑是否如你預期
    img0_path = os.path.join(path, 'input', name, 'im1.png')
    img1_path = os.path.join(path, 'target', name, 'im2.png')
    img2_path = os.path.join(path, 'input', name, 'im3.png')

    I0 = cv2.imread(img0_path)
    I1 = cv2.imread(img1_path)
    I2 = cv2.imread(img2_path)

    # --- 加入檢查機制 ---
    if I0 is None or I1 is None or I2 is None:
        print(f"警告: 無法讀取圖片，跳過此樣本: {name}")
        print(f"檢查路徑: {img0_path}")
        continue
    # ------------------

    # BGR -> RGB 轉換 (注意：cv2 讀進來是 BGR，通常 VFI 模型需要 RGB)
    # 你原本寫 I0.transpose(2, 0, 1) 只改了維度順序，沒有換顏色通道
    I0 = (torch.tensor(I0[:, :, ::-1].copy().transpose(2, 0, 1)).cuda().float() / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2[:, :, ::-1].copy().transpose(2, 0, 1)).cuda().float() / 255.).unsqueeze(0)

    mid = model.inference(I0, I2, TTA=TTA, fast_TTA=TTA)[0]

    # I1 同樣需要處理
    I1_tensor = torch.tensor(I1[:, :, ::-1].copy().transpose(2, 0, 1)).cuda().float().unsqueeze(0) / 255.
    ssim = ssim_matlab(I1_tensor, mid.unsqueeze(0)).detach().cpu().numpy()

    mid = mid.detach().cpu().numpy().transpose(1, 2, 0) 
    I1_norm = I1[:, :, ::-1].copy() / 255. # 轉為 RGB 且正規化以計算 PSNR
    
    psnr = -10 * math.log10(((I1_norm - mid) * (I1_norm - mid)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    print("Avg PSNR: {:.4f} SSIM: {:.4f}".format(np.mean(psnr_list), np.mean(ssim_list)))