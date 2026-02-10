import cv2
import os
import torch
import numpy as np
import random
import glob
from torch.utils.data import Dataset, DataLoader
from config import *

# ==============================================================================
# THESIS PHASE 3 IMPLEMENTATION:
# 1. X4KDataset: High-res training with temporal subsampling.
# 2. MixedDataset: Combines Vimeo and X4K with specific ratios.
# ==============================================================================

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class X4KDataset(Dataset):
    def __init__(self, dataset_name, path, stride_range=(8, 32)):
        self.dataset_name = dataset_name
        self.path = path
        self.stride_range = stride_range
        self.h = 256 # Target patch height
        self.w = 256 # Target patch width
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"X4K dataset path not found: {path}")
        
        # X4K Structure: path/train/SCENE/*.png
        search_path = os.path.join(path, 'train', '*')
        self.scenes = sorted(glob.glob(search_path))
        if len(self.scenes) == 0:
            raise RuntimeError(f"X4KDataset: No scenes found in {os.path.join(path, 'train')}. Check folder structure.")
        print(f"X4KDataset: Found {len(self.scenes)} scenes in {path}")

    def __len__(self):
        return len(self.scenes)

    def getimg(self, index, _depth=0):
        if _depth > 10:
            raise RuntimeError("X4KDataset: Too many retries finding valid scene")
        scene_path = self.scenes[index]
        frames = sorted(glob.glob(os.path.join(scene_path, '*.png')))
        
        if len(frames) < self.stride_range[0] + 1:
            # Fallback if scene too short
            return self.getimg(random.randint(0, len(self.scenes)-1), _depth + 1)
            
        # Temporal Subsampling (Phase 3 Requirement)
        max_stride = min(self.stride_range[1], len(frames) - 1)
        stride = random.randint(self.stride_range[0], max_stride)
        # Sample t, t+stride/2, t+stride
        t = random.randint(0, len(frames) - stride - 1)
        
        img0 = cv2.imread(frames[t])
        gt = cv2.imread(frames[t + stride // 2])
        img1 = cv2.imread(frames[t + stride])
        
        return img0, gt, img1

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        # Resize if image is smaller than target crop
        if ih < h or iw < w:
            scale = max(h / ih, w / iw)
            new_h, new_w = int(ih * scale) + 1, int(iw * scale) + 1
            img0 = cv2.resize(img0, (new_w, new_h))
            gt = cv2.resize(gt, (new_w, new_h))
            img1 = cv2.resize(img1, (new_w, new_h))
            ih, iw = new_h, new_w
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0, gt, img1 = self.aug(img0, gt, img1, self.h, self.w)
        
        # Standard augmentations
        if random.uniform(0, 1) < 0.5:
            img0, img1 = img1, img0
        if random.uniform(0, 1) < 0.5:
            img0 = img0[::-1]
            img1 = img1[::-1]
            gt = gt[::-1]
        if random.uniform(0, 1) < 0.5:
            img0 = img0[:, ::-1]
            img1 = img1[:, ::-1]
            gt = gt[:, ::-1]

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)

class MixedDataset(Dataset):
    def __init__(self, vimeo_dataset, x4k_dataset, ratio=(2, 1)):
        """
        ratio: (vimeo_weight, x4k_weight). 
        e.g., (2, 1) means Vimeo is sampled twice as often.
        """
        self.vimeo = vimeo_dataset
        self.x4k = x4k_dataset
        self.ratio = ratio
        
        # Calculate virtual length
        self.v_len = len(vimeo_dataset)
        self.x_len = len(x4k_dataset)
        
        # Total weight
        self.total_weight = sum(ratio)

    def __len__(self):
        # Use max of weighted lengths to ensure all data is covered
        v_w, x_w = self.ratio
        return max(self.v_len, int(self.x_len * v_w / max(x_w, 1)))

    def __getitem__(self, index):
        # Sample based on ratio
        v_weight, x_weight = self.ratio
        if v_weight == 0:
            return self.x4k[random.randint(0, self.x_len - 1)]
        elif x_weight == 0:
            return self.vimeo[random.randint(0, self.v_len - 1)]
        elif random.random() < (x_weight / self.total_weight):
            return self.x4k[random.randint(0, self.x_len - 1)]
        else:
            return self.vimeo[random.randint(0, self.v_len - 1)]

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, path, batch_size=32, crop_size=None):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = crop_size[0] if crop_size else 256
        self.w = crop_size[1] if crop_size else 448
        self.data_root = path
        self.image_root = os.path.join(self.data_root, 'sequences')
        
        # Support both 'triplet' and 'septuplet' naming
        train_fn_tri = os.path.join(self.data_root, 'tri_trainlist.txt')
        train_fn_sep = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn_tri = os.path.join(self.data_root, 'tri_testlist.txt')
        test_fn_sep = os.path.join(self.data_root, 'sep_testlist.txt')
        
        if os.path.exists(train_fn_sep):
            self.train_fn = train_fn_sep
            self.test_fn = test_fn_sep
            self.is_septuplet = True
        else:
            self.train_fn = train_fn_tri
            self.test_fn = test_fn_tri
            self.is_septuplet = False
            
        with open(self.train_fn, 'r') as f:
            self.trainlist = [l for l in f.read().splitlines() if l.strip()]
        with open(self.test_fn, 'r') as f:
            self.testlist = [l for l in f.read().splitlines() if l.strip()]
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name != 'test':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        if self.is_septuplet:
            # For septuplet, we sample im1, im4, im7 as a triplet (standard VFI practice)
            imgpaths = [imgpath + '/im1.png', imgpath + '/im4.png', imgpath + '/im7.png']
        else:
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1
            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index)
                
        if 'train' in self.dataset_name:
            img0, gt, img1 = self.aug(img0, gt, img1, self.h, self.w)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img1, img0 = img0, img1
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]

            p = random.uniform(0, 1)
            if p < 0.25:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)
