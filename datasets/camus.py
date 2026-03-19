class CAMUSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, patient_list=None):
        all_image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and '_mask' not in f])
        
        patient_ids = sorted(list(set([f.split('_')[0] for f in all_image_files])))
        
        if patient_list is not None:
            selected = set(patient_list)
            all_image_files = [f for f in all_image_files if f.split('_')[0] in selected]
        
        self.image_files = all_image_files
        self.mask_files = [f.replace('.png', '_mask.png') for f in all_image_files]
        
        assert len(self.image_files) == len(self.mask_files),
        
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
        mask = (mask > 128).astype(np.uint8)
        
        img_file = self.image_files[idx]
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        
        return image, mask, img_file
