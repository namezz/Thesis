from time import time
import sys
import torch
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)

'''==========import from our code=========='''
sys.path.append('.')
import config as cfg
from Trainer import Model

# Resolution presets
RESOLUTION_PRESETS = {
    '720p': (720, 1280),
    '1080p': (1080, 1920),
    '2k': (1080, 2048),
    '2160p': (2160, 3840),  # 4K
    '4k': (2160, 3840),
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='hybrid_v1_baseline', type=str)
parser.add_argument('--H', default=None, type=int, help='Height (overrides resolution preset)')
parser.add_argument('--W', default=None, type=int, help='Width (overrides resolution preset)')
parser.add_argument('--resolution', default='720p', type=str, 
                    choices=['720p', '1080p', '2k', '2160p', '4k'],
                    help='Resolution preset')
parser.add_argument('--warmup', default=50, type=int, help='Warmup iterations')
parser.add_argument('--iterations', default=100, type=int, help='Test iterations')
args = parser.parse_args()

'''==========Model setting=========='''
TTA = False
cfg.MODEL_CONFIG['LOGNAME'] = args.model

model = Model(-1)
model.load_model()
model.eval()
model.device()

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# Determine resolution
if args.H is not None and args.W is not None:
    H, W = args.H, args.W
else:
    H, W = RESOLUTION_PRESETS[args.resolution]

I0 = torch.rand(1, 3, H, W).cuda()
I1 = torch.rand(1, 3, H, W).cuda()

print(f'=== Inference Time Test ===')
print(f'Model: {model.name}')
print(f'Resolution: {args.resolution} ({H}x{W})')
print(f'TTA: {TTA}')
print(f'Warmup: {args.warmup} iterations')
print(f'Test: {args.iterations} iterations')
print('===========================')

with torch.no_grad():
    # Warmup
    for i in range(args.warmup):
        pred = model.inference(I0, I1)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    time_stamp = time()
    for i in range(args.iterations):
        pred = model.inference(I0, I1)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    avg_time_ms = (time() - time_stamp) / args.iterations * 1000
    fps = 1000 / avg_time_ms
    
    print(f'Average time: {avg_time_ms:.2f} ms')
    print(f'FPS: {fps:.2f}')