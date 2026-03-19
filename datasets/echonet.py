import os
import cv2
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, Sampler
from albumentations.pytorch import ToTensorV2

class EchoNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, filelist_csv_path=None, max_samples=None):

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        if max_samples is not None:
            image_files = image_files[:max_samples]
            mask_files = mask_files[:max_samples]
        
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.volumes = []
        self.efs = []
        self.df = None
        if filelist_csv_path and os.path.exists(filelist_csv_path):
            self.df = pd.read_csv(filelist_csv_path)
            for mask_file in mask_files:
                video_id = mask_file.split('_')[0] 
                frame_type = mask_file.split('_')[1]
                row = self.df[self.df['FileName'].str.replace('.avi', '') == video_id]
                if not row.empty:
                    edv = row['EDV'].values[0]
                    esv = row['ESV'].values[0]
                    ef = row['EF'].values[0]
                    volume = edv if frame_type.upper() == 'ED' else esv
                else:
                    volume = 0
                    ef = 0
                self.volumes.append(volume)
                self.efs.append(ef)
        else:
            self.volumes = [0] * len(mask_files)
            self.efs = [0] * len(mask_files)
        
        self.buckets = self._compute_buckets()

    def _compute_buckets(self):
        small_volume_threshold_ed = 90
        small_volume_threshold_es = 40
        high_ef_threshold = 55
        buckets = {'small_low': [], 'small_high': [], 'other': []}
        for i in range(len(self.volumes)):
            frame_type = self.mask_files[i].split('_')[1].upper()
            volume_threshold = small_volume_threshold_ed if frame_type == 'ED' else small_volume_threshold_es
            if self.volumes[i] < volume_threshold:
                if self.efs[i] > high_ef_threshold:
                    buckets['small_high'].append(i)
                else:
                    buckets['small_low'].append(i)
            else:
                buckets['other'].append(i)
        return buckets

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask, self.image_files[idx]


class BalancedSampler(Sampler):
    def __init__(self, dataset, small_ratio=0.15):
        self.dataset = dataset
        self.buckets = dataset.buckets
        self.small_ratio = small_ratio
        self.num_samples = len(dataset)
    
    def __iter__(self):
        small_indices = self.buckets['small_low'] + self.buckets['small_high']
        other_indices = self.buckets['other']
        
        indices = []
        small_per_batch = int(self.num_samples * self.small_ratio)
        
        while len(indices) < self.num_samples:
            small_sample = random.sample(small_indices, min(small_per_batch, len(small_indices)))
            other_sample = random.sample(other_indices, min(self.num_samples - len(indices), len(other_indices)))
            indices.extend(small_sample + other_sample)
        
        random.shuffle(indices)
        return iter(indices[:self.num_samples])
    
    def __len__(self):
        return self.num_samples
