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
from benchmark.utils.padder import InputPadder
from benchmark.utils.pytorch_msssim import ssim_matlab

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='hybrid_v1_baseline', type=str)
parser.add_argument('--path', type=str, required=True, help="Path to SNU-FILM dataset")
args = parser.parse_args()

'''==========Model setting=========='''
TTA = False
down_scale = 0.5
if args.model == 'hybrid_v1_baseline':
    cfg.MODEL_CONFIG['LOGNAME'] = 'hybrid_v1_baseline'
else:
    cfg.MODEL_CONFIG['LOGNAME'] = args.model

model = Model(-1)
model.load_model()
model.eval()
model.device()

# --- Helper Functions 整合 ---

def _resolve_frame_path(base_path, relative_path):
    """處理 SNU-FILM 資料集路徑不一致的問題"""
    # 1. 嘗試原始路徑
    p = os.path.join(base_path, relative_path)
    if os.path.exists(p):
        return p
    
    # 2. 處理常見的前綴與命名不一致問題
    parts = relative_path.split('/')
    if len(parts) > 2 and parts[0] == 'data' and parts[1] == 'SNU-FILM':
        # 移除 'data/SNU-FILM/' 前綴
        stripped_path = os.path.join(base_path, *parts[2:])
        if os.path.exists(stripped_path):
            return stripped_path
             
        # 處理 GOPRO_test vs GOPRO_test_ 的資料夾命名差異
        if 'GOPRO_test' in parts:
            new_parts = [p if p != 'GOPRO_test' else 'GOPRO_test_' for p in parts[2:]]
            gopro_path = os.path.join(base_path, *new_parts)
            if os.path.exists(gopro_path):
                return gopro_path
    return p

def _ensure_image(path_str):
    """確保圖片正確讀取，避免 NoneType 錯誤"""
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"找不到檔案: {path_str}")
    img = cv2.imread(path_str)
    if img is None:
        raise RuntimeError(f"OpenCV 無法讀取圖片: {path_str}。請檢查檔案格式是否損壞。")
    return img

# --- Testing Loop ---

print(f'=========================Starting testing=========================')
print(f'Dataset: SNU_FILM   Model: {model.name}   TTA: {TTA}')

path = args.path
level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt'] 

for test_file in level_list:
    psnr_list, ssim_list = [], []
    file_list = []
    
    txt_path = os.path.join(path, test_file)
    if not os.path.exists(txt_path):
        print(f"跳過 {test_file}: 找不到標籤檔")
        continue

    with open(txt_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                warnings.warn(f"跳過格式錯誤的行: '{line}'")
                continue
            file_list.append(parts[:3])

    print(f'Testing level: {test_file[:-4]}')
    for line in tqdm(file_list):
        try:
            # 解析路徑並讀取圖片
            I0_path = _resolve_frame_path(path, line[0])
            I1_path = _resolve_frame_path(path, line[1])
            I2_path = _resolve_frame_path(path, line[2])
            
            I0_raw = _ensure_image(I0_path)
            I1_raw = _ensure_image(I1_path)
            I2_raw = _ensure_image(I2_path)

            # 轉換為 Tensor (注意：如果模型是 RGB，建議加 [:, :, ::-1])
            I0 = (torch.tensor(I0_raw.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
            I1 = (torch.tensor(I1_raw.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
            I2 = (torch.tensor(I2_raw.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

            # Padding 處理
            padder = InputPadder(I0.shape, divisor=32)
            I0_padded, I2_padded = padder.pad(I0, I2)

            # 模型推理 (這裡使用 hr_inference 或 inference 取決於你的 Model 類別定義)
            # 參考你提供的代碼使用 model.hr_inference
            I1_pred = model.hr_inference(I0_padded, I2_padded, TTA, down_scale=down_scale, fast_TTA=TTA)[0]
            I1_pred = padder.unpad(I1_pred)

            # 計算 SSIM
            ssim = ssim_matlab(I1, I1_pred.unsqueeze(0)).detach().cpu().numpy()

            # 計算 PSNR (BGR 空間或轉換後空間需統一)
            I1_pred_np = I1_pred.detach().cpu().numpy().transpose(1, 2, 0)   
            I1_target_np = I1_raw / 255.0
            
            psnr = -10 * math.log10(((I1_target_np - I1_pred_np) ** 2).mean())
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
        except Exception as e:
            print(f"處理樣本時出錯 {line}: {e}")
            continue
    
    if psnr_list:
        print('Avg PSNR: {:.4f} SSIM: {:.4f}'.format(np.mean(psnr_list), np.mean(ssim_list)))
    else:
        print(f'Level {test_file} 無有效測試結果。')