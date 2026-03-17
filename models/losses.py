import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


sobel_kernel_x = torch.tensor([[[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]]], dtype=torch.float32)
sobel_kernel_y = torch.tensor([[[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]]], dtype=torch.float32)

def boundary_loss(pred_prob, target):
    pred_dx = F.conv2d(pred_prob, sobel_kernel_x.to(pred_prob.device), padding=1)
    pred_dy = F.conv2d(pred_prob, sobel_kernel_y.to(pred_prob.device), padding=1)
    target_dx = F.conv2d(target, sobel_kernel_x.to(target.device), padding=1)
    target_dy = F.conv2d(target, sobel_kernel_y.to(target.device), padding=1)
    
    pred_mag = torch.sqrt(pred_dx**2 + pred_dy**2 + 1e-8)
    target_mag = torch.sqrt(target_dx**2 + target_dy**2 + 1e-8)
    
    return F.mse_loss(pred_mag, target_mag)

class SegLoss(nn.Module):
    def __init__(self, gamma=2.0, focal_weight=0.5, smooth=1e-8,
                 use_boundary=False, boundary_weight=0.05):
        super().__init__()
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        # pred: logits (B, C, H, W)，target: (B, C, H, W) 二值0/1
        pred_prob = torch.sigmoid(pred.float())
        
        intersection = (pred_prob * target).sum(dim=(2, 3))  # (B, C)
        union = pred_prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff.mean() 
        
        pos_mask = target == 1.0
        focal_term = torch.zeros_like(pred_prob)
        if pos_mask.any():
            log_prob = torch.log(pred_prob[pos_mask] + 1e-8)
            focal_term[pos_mask] = ((1 - pred_prob[pos_mask]) ** self.gamma) * log_prob
        focal_loss = -focal_term.sum() / (pos_mask.sum() + self.smooth)  # mean over positive pixels
        
        total_loss = dice_loss + self.focal_weight * focal_loss
        
        if self.use_boundary:
            bound_loss = boundary_loss(pred_prob, target)
            total_loss = total_loss + self.boundary_weight * bound_loss
        
        return total_loss
