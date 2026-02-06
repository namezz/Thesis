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
parser.add_argument('--path', type=str, required=True, help="Path to MiddleBury OTHER dataset")
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
print(f'Dataset: MiddleBury-Other   Model: {model.name}   TTA: {TTA}')

path = args.path

# MiddleBury OTHER dataset structure:
# other-data/: Input frames (frame10.png, frame11.png)
# other-gt-interp/: Ground truth interpolated frames
# Scenes: Beanbags, Dimetrodon, DogDance, Grove2, Grove3, Hydrangea, MiniCooper, RubberWhale, Urban2, Urban3, Venus, Walking

scenes = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 
          'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 
          'Venus', 'Walking']

ie_list = []  # Interpolation Error

for scene in tqdm(scenes, desc="Processing scenes"):
    # Check different possible path structures
    data_path = os.path.join(path, 'other-data', scene)
    gt_path = os.path.join(path, 'other-gt-interp', scene)
    
    # Alternative structure (flat)
    if not os.path.exists(data_path):
        data_path = os.path.join(path, scene)
        gt_path = os.path.join(path, scene)
    
    # Load input frames
    img0_path = os.path.join(data_path, 'frame10.png')
    img1_path = os.path.join(data_path, 'frame11.png')
    gt_path_file = os.path.join(gt_path, 'frame10i11.png')
    
    if not os.path.exists(img0_path):
        print(f"Warning: Skipping {scene} - files not found")
        continue
    
    I0 = cv2.imread(img0_path)
    I1 = cv2.imread(img1_path)
    
    if os.path.exists(gt_path_file):
        GT = cv2.imread(gt_path_file)
    else:
        print(f"Warning: GT not found for {scene}")
        continue
    
    if I0 is None or I1 is None or GT is None:
        print(f"Warning: Failed to read images for {scene}")
        continue
    
    # Convert to tensor (Keep BGR, consistent with training; HWC -> CHW)
    I0_tensor = torch.from_numpy(I0.transpose(2, 0, 1).copy()).float().unsqueeze(0).cuda() / 255.
    I1_tensor = torch.from_numpy(I1.transpose(2, 0, 1).copy()).float().unsqueeze(0).cuda() / 255.
    GT_tensor = torch.from_numpy(GT.transpose(2, 0, 1).copy()).float().unsqueeze(0).cuda() / 255.
    
    # Padding
    padder = InputPadder(I0_tensor.shape, divisor=32)
    I0_padded, I1_padded = padder.pad(I0_tensor, I1_tensor)
    
    # Inference
    pred = model.inference(I0_padded, I1_padded, TTA=TTA)[0]
    pred = padder.unpad(pred)
    
    # Calculate Interpolation Error (IE)
    # IE = sqrt(mean((pred - GT)^2)) * 255
    diff = (pred - GT_tensor) * 255
    ie = torch.sqrt((diff ** 2).mean()).item()
    ie_list.append(ie)
    
print(f'\n=========================Results=========================')
print(f'MiddleBury-Other  IE: {np.mean(ie_list):.3f}')
print(f'=========================================================')

# Per-scene results
print(f'\nPer-scene IE:')
for scene, ie in zip(scenes[:len(ie_list)], ie_list):
    print(f'  {scene}: {ie:.3f}')
