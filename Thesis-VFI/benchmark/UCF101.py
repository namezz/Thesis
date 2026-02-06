import os
import sys
import cv2
import math
import torch
import argparse
import warnings
import numpy as np
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
parser.add_argument('--path', type=str, required=True, help="Path to UCF101 dataset")
args = parser.parse_args()
'''==========Model setting=========='''
TTA = True
cfg.MODEL_CONFIG['LOGNAME'] = args.model

model = Model(-1)
model.load_model()
model.eval()
model.device()

# --- Helper Function ---
def load_img_tensor(path):
    """讀取圖片並轉換為 Tensor，包含錯誤檢查"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到檔案: {path}")
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"無法讀取圖片內容: {path}")
    # Keep BGR (consistent with training data which uses cv2.imread BGR)
    return torch.from_numpy(img.transpose(2, 0, 1).copy()).float().cuda().unsqueeze(0) / 255.0

print(f'=========================Starting testing=========================')
print(f'Dataset: UCF101   Model: {model.name}   TTA: {TTA}')

path = args.path
# 只讀取資料夾，過濾掉隱藏檔與非資料夾檔案
dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
psnr_list, ssim_list = [], []

for d in tqdm(dirs):
    try:
        # 設定檔案路徑
        img0_path = os.path.join(path, d, 'frame_00.png')
        img1_path = os.path.join(path, d, 'frame_02.png')
        gt_path   = os.path.join(path, d, 'frame_01_gt.png')

        # 讀取 Tensor
        img0 = load_img_tensor(img0_path)
        img1 = load_img_tensor(img1_path)
        gt_tensor = load_img_tensor(gt_path)

        # 模型推理
        # 模型輸出通常是 [C, H, W] 的 Tensor
        pred = model.inference(img0, img1, TTA=TTA, fast_TTA=TTA)[0]

        # 計算 SSIM (依照原程式邏輯進行 round 處理)
        # pred shape: (1, C, H, W), gt_tensor shape: (1, C, H, W)
        pred_clamped = torch.clamp(pred, 0, 1)
        pred_rounded = torch.round(pred_clamped * 255) / 255.
        ssim = ssim_matlab(gt_tensor, pred_rounded).detach().cpu().numpy()

        # 計算 PSNR
        out_np = pred_rounded[0].detach().cpu().numpy().transpose(1, 2, 0)
        gt_np = gt_tensor[0].cpu().numpy().transpose(1, 2, 0)
        
        # 避免溢位與計算 MSE
        mse = np.mean((gt_np - out_np) ** 2)
        if mse == 0:
            psnr = 100 # 完全相同
        else:
            psnr = -10 * math.log10(mse)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    except Exception as e:
        print(f"跳過資料夾 {d}，錯誤原因: {e}")
        continue

if psnr_list:
    print("\n測試完成！")
    print("Avg PSNR: {:.4f} SSIM: {:.4f}".format(np.mean(psnr_list), np.mean(ssim_list)))
else:
    print("\n錯誤：沒有成功的測試樣本，請檢查資料集路徑。")