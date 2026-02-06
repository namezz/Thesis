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
parser.add_argument('--model', default='hybrid_v1_baseline', type=str)
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

'''==========Model setting=========='''
TTA = False
if args.model == 'hybrid_v1_baseline':
    cfg.MODEL_CONFIG['LOGNAME'] = 'hybrid_v1_baseline'
else:
    cfg.MODEL_CONFIG['LOGNAME'] = args.model

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

    # Keep BGR (consistent with training data which uses cv2.imread BGR)
    I0 = (torch.tensor(I0.transpose(2, 0, 1).copy()).cuda().float() / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1).copy()).cuda().float() / 255.).unsqueeze(0)

    mid = model.inference(I0, I2, TTA=TTA, fast_TTA=TTA)[0]

    # I1 同樣保持 BGR
    I1_tensor = torch.tensor(I1.transpose(2, 0, 1).copy()).cuda().float().unsqueeze(0) / 255.
    ssim = ssim_matlab(I1_tensor, mid).detach().cpu().numpy()

    mid_np = mid[0].detach().cpu().numpy().transpose(1, 2, 0) 
    I1_norm = I1 / 255.  # BGR, consistent with model output
    
    psnr = -10 * math.log10(((I1_norm - mid_np) * (I1_norm - mid_np)).mean())
    psnr_list.append(psnr)
    ssim_list.append(ssim)

    print("Avg PSNR: {:.4f} SSIM: {:.4f}".format(np.mean(psnr_list), np.mean(ssim_list)))