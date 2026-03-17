import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from torch.amp import autocast, GradScaler
import random
import shutil
from peft import LoraConfig, get_peft_model

if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    sam_checkpoint = "/kaggle/input/segment-anything/pytorch/vit-b/1/model.pth" 
    model_type = "vit_b" 
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
    sam.to(device=device) 
    predictor = SamPredictor(sam) 
    prompt_generator = ViTPromptGenerator(sam) 
    for param in prompt_generator.feature_extractor.image_encoder.parameters(): 
        param.requires_grad = False
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        prompt_generator = nn.DataParallel(prompt_generator)
    prompt_generator.to(device)

    
    all_patients = sorted(list(set([f.split('_')[0] for f in os.listdir(camus_image_dir) if f.endswith('.png') and '_mask' not in f])))
    random.shuffle(all_patients)
    n = len(all_patients)
    train_patients = all_patients[:int(0.8 * n)]
    val_patients = all_patients[int(0.8 * n):int(0.9 * n)]
    test_patients = all_patients[int(0.9 * n):]
    
    train_dataset = CAMUSDataset(camus_image_dir, camus_mask_dir, transform=train_transform, patient_list=train_patients)
    val_dataset = CAMUSDataset(camus_image_dir, camus_mask_dir, transform=val_transform, patient_list=val_patients)
    test_dataset = CAMUSDataset(camus_image_dir, camus_mask_dir, transform=val_transform, patient_list=test_patients)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    total_epochs = 100 

    best_model_path = train_prompt_generator( 
        prompt_generator, train_loader, val_loader, 
        epochs=80, lr=1e-4, device=device, patience=5, 
        checkpoint_path='/kaggle/working/output/checkpoint.pth',  
        start_epoch=0
    ) 

    checkpoint_file = os.path.join(output_dir, 'checkpoint.pth') 
    if os.path.exists(checkpoint_file): 
        checkpoint = torch.load(checkpoint_file) 
        current_epoch = checkpoint['epoch'] 
        if current_epoch >= total_epochs:
            print("Loading best model for testing...")
            checkpoint = torch.load(best_model_path, map_location=device)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
            prompt_generator = ViTPromptGenerator(sam_model=sam).to(device)
                
            prompt_generator.load_state_dict(new_state_dict, strict=False)  # strict=False防止删neck等小改动报错
            prompt_generator.eval()
                
            print(f"Best model loaded! Val Dice: {checkpoint.get('val_dice', 'N/A'):.5f}")
            metrics = {'dice': [], 'iou': [], 'hd': [], 'hd95': []} 
            inference_times = [] 
            for idx, (image, mask, img_file) in enumerate(test_loader): 
                image = image.to(device)
                mask = mask.to(device).float()
                img_file = img_file[0]  # str             
                try: 
                    start_time = time.perf_counter() 
                    visualize = (idx < 30) 
                    result = process_and_visualize(
                        image_tensor=image, 
                        mask_tensor=mask, 
                        output_dir=output_dir, 
                        img_file=img_file,
                        prompt_generator=prompt_generator, 
                        sam_predictor=predictor, 
                        visualize=visualize,
                        device=device
                    )   
                    end_time = time.perf_counter() 
                    inference_time = end_time - start_time 
                    inference_times.append(inference_time) 
                    for key in metrics: 
                        metrics[key].append(result[key]) 
                except Exception as e: 
                    print(f"Error processing {img_file}: {str(e)}")
                    
            if metrics['dice']: 
                print("\nAverage Metrics on Test Set (All Samples, Auto Prompt):") 
                print(f"Dice Score: {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}") 
                print(f"IoU: {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}") 
                print(f"Hausdorff Distance: {np.mean(metrics['hd']):.4f} ± {np.std(metrics['hd']):.4f}") 
                print(f"95% Hausdorff Distance: {np.mean(metrics['hd95']):.4f} ± {np.std(metrics['hd95']):.4f}") 
                avg_inference_time = np.mean(inference_times) 
                print(f"Average Inference Time per Sample: {avg_inference_time:.4f} seconds") 
            else: 
                print("No images processed successfully.") 
        else: 
            print(f"Training not complete (current epoch: {current_epoch}/{total_epochs}). Skipping test and plots.") 
    else: 
        print("No checkpoint found. Skipping test and plots.")
