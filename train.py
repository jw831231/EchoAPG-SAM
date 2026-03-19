import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2


from models.sam_adapter import EnhancedSAM
from models.prompt_generator import ViTPromptGenerator
from models.losses import SegLoss 
from datasets.camus import CAMUSDataset


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "outputs"
checkpoint_dir = "checkpoints"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


train_transform = A.Compose([
    A.Rotate(limit=30, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ToTensorV2()
])


print("正在构建 LoRA + MSPAd + 智能Prompt Generator 模型...")
enhanced_sam = EnhancedSAM(
    model_type=cfg["model"]["type"],
    checkpoint=cfg["model"]["checkpoint"],
    lora_r=cfg["model"]["lora_r"]
).to(device)

model = ViTPromptGenerator(enhanced_sam.sam).to(device)


for name, param in model.named_parameters():
    if "lora" in name.lower() or "mspad" in name.lower():
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable_params:,} (LoRA + MSPAd + Prompt Generator)")


print(" 加载 CAMUS 数据集（患者级 8:1:1 划分）...")
all_patients = sorted(list(set([f.split('_')[0] for f in os.listdir(cfg["data"]["image_dir"]) 
                               if f.endswith('.png') and '_mask' not in f])))
random.shuffle(all_patients)
n = len(all_patients)

train_patients = all_patients[:int(cfg["data"]["train_ratio"] * n)]
val_patients   = all_patients[int(cfg["data"]["train_ratio"] * n):int(0.9 * n)]

train_dataset = CAMUSDataset(
    image_dir=cfg["data"]["image_dir"],
    mask_dir=cfg["data"]["mask_dir"],
    transform=train_transform,
    patient_list=train_patients
)
val_dataset = CAMUSDataset(
    image_dir=cfg["data"]["image_dir"],
    mask_dir=cfg["data"]["mask_dir"],
    transform=val_transform,
    patient_list=val_patients
)

train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2, pin_memory=True)


criterion = SegLoss(
    gamma=2.0,
    focal_weight=cfg["loss"]["focal_weight"],
    boundary_weight=cfg["loss"]["boundary_weight"]
).to(device)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["training"]["lr"])
scaler = GradScaler(device='cuda')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)


start_epoch = 0
best_val_dice = -1.0
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
best_model_path = os.path.join(checkpoint_dir, "best_lora_mspad_autoprompt.pth")

if os.path.exists(checkpoint_path):
    print(f" 恢复断点训练: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch']
    best_val_dice = ckpt.get('best_val_dice', -1.0)
    print(f"从 epoch {start_epoch} 继续训练")


def compute_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * intersection) / (union + 1e-8)
    iou = intersection / (union - intersection + 1e-8)
    return dice.mean().item(), iou.mean().item()


print("开始训练...")
train_losses, val_losses = [], []
train_dices, val_dices = [], []
train_ious, val_ious = [], []

for epoch in range(start_epoch, cfg["training"]["epochs"]):
    model.train()
    train_loss = 0.0
    train_dice_total = 0.0
    train_iou_total = 0.0
    num_batches = 0

    for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        images, masks = images.to(device), masks.to(device).float()
        
        
        images_1024 = F.interpolate(images, size=(1024, 1024), mode='bicubic', align_corners=False)
        images_1024 = images_1024.repeat(1, 3, 1, 1)
        images_1024 = images_1024 * 255.0
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1,3,1,1)
        std  = torch.tensor([58.395, 57.12, 57.375], device=device).view(1,3,1,1)
        images_1024 = (images_1024 - mean) / std
        
        masks_1024 = F.interpolate(masks, size=(1024, 1024), mode='nearest')

        with autocast(device_type='cuda'):
            pred = model(images_1024)
            loss = criterion(pred, masks_1024)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        dice, iou = compute_metrics(pred, masks_1024)
        train_dice_total += dice
        train_iou_total += iou
        num_batches += 1

    train_loss /= num_batches
    train_dice = train_dice_total / num_batches
    train_iou = train_iou_total / num_batches

    train_losses.append(train_loss)
    train_dices.append(train_dice)
    train_ious.append(train_iou)

    model.eval()
    val_loss = 0.0
    val_dice_total = 0.0
    val_iou_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            images, masks = images.to(device), masks.to(device).float()
            
            images_1024 = F.interpolate(images, size=(1024, 1024), mode='bicubic', align_corners=False)
            images_1024 = images_1024.repeat(1, 3, 1, 1)
            images_1024 = images_1024 * 255.0
            mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1,3,1,1)
            std  = torch.tensor([58.395, 57.12, 57.375], device=device).view(1,3,1,1)
            images_1024 = (images_1024 - mean) / std
            
            masks_1024 = F.interpolate(masks, size=(1024, 1024), mode='nearest')

            with autocast(device_type='cuda'):
                pred = model(images_1024)
                loss = criterion(pred, masks_1024)

            val_loss += loss.item()
            dice, iou = compute_metrics(pred, masks_1024)
            val_dice_total += dice
            val_iou_total += iou
            num_batches += 1

    val_loss /= num_batches
    val_dice = val_dice_total / num_batches
    val_iou = val_iou_total / num_batches

    val_losses.append(val_loss)
    val_dices.append(val_dice)
    val_ious.append(val_iou)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:3d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} "
          f"| Train Dice {train_dice:.4f} | Val Dice {val_dice:.4f} | LR {current_lr:.6f}")

    scheduler.step(val_dice)

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_dice': best_val_dice,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices,
        'train_ious': train_ious,
        'val_ious': val_ious,
    }, checkpoint_path)

    if val_dice > best_val_dice + 0.0005:
        best_val_dice = val_dice
        torch.save(model.state_dict(), best_model_path)
        enhanced_sam.save_clean_state(os.path.join(checkpoint_dir, "clean_best.pth"))
        print(f" 新最佳模型！Val Dice: {val_dice:.5f}")


epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_dices, label='Train Dice')
plt.plot(epochs_range, val_dices, label='Val Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.title('Epochs vs Dice')
plt.legend()
plt.savefig(os.path.join(output_dir, 'dice_curve.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_ious, label='Train IoU')
plt.plot(epochs_range, val_ious, label='Val IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.title('Epochs vs IoU')
plt.legend()
plt.savefig(os.path.join(output_dir, 'iou_curve.png'))
plt.close()

print(f"训练完成！")
print(f"最佳模型保存至: {best_model_path}")
print(f"损失/指标曲线保存至: {output_dir}")
