import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class EchoNetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

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
