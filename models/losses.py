import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

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
