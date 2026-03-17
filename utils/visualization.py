import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

def get_euclidean_point(mask):
    dist_transform = distance_transform_edt(mask)
    if np.max(dist_transform) == 0:
        return (mask.shape[1] // 2, mask.shape[0] // 2)
    y, x = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    return (int(x), int(y))

def get_bounding_box(mask):
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        size = min(mask.shape) // 4
        return [size, size, mask.shape[1]-size, mask.shape[0]-size]
    return [np.min(x), np.min(y), np.max(x), np.max(y)]

def dice_coefficient(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return 2 * intersection / (union + 1e-8) if union > 0 else 0.0

def iou_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8) if union > 0 else 0.0

def hausdorff_distance(pred, gt):
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('inf')
    pred_coords = np.argwhere(pred > 0)
    gt_coords = np.argwhere(gt > 0)
    hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
    return max(hd1, hd2)

def hausdorff_distance_95(pred, gt):
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('inf')
    pred_coords = np.argwhere(pred > 0)
    gt_coords = np.argwhere(gt > 0)
    dist_matrix = np.sqrt(np.sum((pred_coords[:, np.newaxis, :] - gt_coords[np.newaxis, :, :])**2, axis=2))
    hd1 = np.percentile(np.min(dist_matrix, axis=1), 95)
    hd2 = np.percentile(np.min(dist_matrix, axis=0), 95)
    return max(hd1, hd2)

def process_and_visualize(image_tensor, mask_tensor, output_dir, img_file, prompt_generator, sam_predictor, visualize=False, device='cuda'):
    # image_tensor: [1, 1, 256, 256] or [3, 256, 256] after repeat
    # mask_tensor: [1, 1, 256, 256] ground truth
    
    image_tensor = image_tensor.squeeze(0)  # [1 or 3, 256, 256]
    if image_tensor.shape[0] == 1:
        image = image_tensor[0].cpu().numpy()  # grayscale to numpy
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        image_rgb = (image_rgb * 255).astype(np.uint8)  # if normalized
    
    gt_mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()  # [256, 256]
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    prompt_generator.eval() 
    with torch.no_grad(): 

        input_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 1, 256, 256]
        input_tensor_1024 = F.interpolate(input_tensor, size=(1024, 1024), mode='bicubic', align_corners=False)
        input_tensor_1024 = input_tensor_1024.repeat(1, 3, 1, 1)

        # === 新增SAM normalize ===
        input_tensor_1024 = input_tensor_1024 * 255.0
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        input_tensor_1024 = (input_tensor_1024 - mean) / std

    
        with autocast(device_type='cuda'):
            rough_mask_logits = prompt_generator(input_tensor_1024)           
            #rough_mask_prob = torch.sigmoid(rough_mask_logits).squeeze(0).squeeze(0).cpu().numpy()
            rough_mask_prob = torch.sigmoid(rough_mask_logits)
            rough_mask_prob = torch.nan_to_num(rough_mask_prob, nan=0.0, posinf=1.0, neginf=0.0)

        rough_mask_prob_256 = F.interpolate(rough_mask_prob, size=(256, 256), mode='bilinear', align_corners=False)
        rough_mask_prob_256 = rough_mask_prob_256.squeeze(0).squeeze(0) 
        rough_mask_np = rough_mask_prob_256.cpu().numpy()
        if np.any(np.isnan(rough_mask_np)):
            print(f"Warning: Still NaN in {img_file}, using zero mask")
            rough_mask_np = np.zeros((256, 256), dtype=np.float32)

        rough_mask = (rough_mask_np > 0.5).astype(np.uint8)
    
        kernel = np.ones((5, 5), np.uint8)
        rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel)
        rough_mask = cv2.dilate(rough_mask, kernel, iterations=1)
        rough_mask = cv2.erode(rough_mask, kernel, iterations=1)
        kernel_ellipse = np.ones((7, 7), np.uint8)
        rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_CLOSE, kernel_ellipse, iterations=2)
        rough_mask = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel_ellipse, iterations=1)             
    box = get_bounding_box(rough_mask) 
    point = get_euclidean_point(rough_mask) 
    mask_input = rough_mask.astype(np.float32) * 30.0 - 10.0 
    mask_input_torch = torch.from_numpy(mask_input).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 256, 256]
    
    sam_predictor.set_image(image_rgb) 
    try: 
        masks, scores, logits = sam_predictor.predict( 
            box=np.array(box), 
            point_coords=np.array([point]), 
            point_labels=np.array([1]), 
            mask_input=mask_input_torch[0], 
            multimask_output=False 
        ) 
    except Exception as e: 
        raise ValueError(f"SAM predict 失败: {str(e)}") 

    print(f"SAM output mask shape: {masks.shape}")  # 应该打印 (1, 256, 256) 或类似
    
    sam_mask = masks[0].astype(np.uint8)

    dice_score = dice_coefficient(sam_mask, gt_mask) 
    iou = iou_score(sam_mask, gt_mask) 
    hd = hausdorff_distance(sam_mask, gt_mask) 
    hd95 = hausdorff_distance_95(sam_mask, gt_mask) 
    if visualize: 
        fig, axes = plt.subplots(1, 5, figsize=(25, 5)) 
        axes[0].imshow(image_rgb) 
        axes[0].set_title("Raw Image") 
        axes[0].axis("off") 
        gt_overlay = image_rgb.copy() 
        gt_mask_colored = np.zeros_like(image_rgb) 
        gt_mask_colored[gt_mask == 1] = [255, 0, 255] 
        cv2.addWeighted(gt_mask_colored, 0.8, gt_overlay, 1, 0, gt_overlay) 
        axes[1].imshow(gt_overlay) 
        axes[1].set_title("Ground Truth") 
        axes[1].axis("off") 
        rough_overlay = image_rgb.copy() 
        rough_mask_colored = np.zeros_like(image_rgb) 
        rough_mask_colored[rough_mask == 1] = [0, 255, 0] 
        cv2.addWeighted(rough_mask_colored, 0.8, rough_overlay, 1, 0, rough_overlay) 
        axes[2].imshow(rough_overlay) 
        axes[2].set_title("Rough Mask (ViT Generated)") 
        axes[2].axis("off") 
        prompt_overlay = image_rgb.copy() 
        prompt_colored = np.zeros_like(image_rgb) 
        prompt_colored[rough_mask == 1] = [255, 0, 255] 
        cv2.addWeighted(prompt_colored, 0.8, prompt_overlay, 1, 0, prompt_overlay) 
        x_min, y_min, x_max, y_max = box 
        cv2.rectangle(prompt_overlay, (x_min, y_min), (x_max, y_max), color=(255, 0, 255), thickness=2) 
        cv2.circle(prompt_overlay, point, radius=5, color=(255, 0, 255), thickness=-1) 
        axes[3].imshow(prompt_overlay) 
        axes[3].set_title("Auto Prompt with Box & Point") 
        axes[3].axis("off") 
        pred_overlay = image_rgb.copy() 
        pred_mask_colored = np.zeros_like(image_rgb) 
        pred_mask_colored[sam_mask == 1] = [255, 0, 255] 
        cv2.addWeighted(pred_mask_colored, 0.8, pred_overlay, 1, 0, pred_overlay) 
        axes[4].imshow(pred_overlay) 
        axes[4].set_title(f"Predict\nDice: {dice_score:.3f}, IoU: {iou:.3f}\nHD: {hd:.1f}, HD95: {hd95:.1f}") 
        axes[4].axis("off") 
        plt.suptitle(f"Result for {img_file}\nAuto Mask & Box & Point Prompt (ViT Generated)", fontsize=16) 
        plt.savefig(os.path.join(output_dir, f"{img_file}_auto_results.png"), bbox_inches="tight") 
        plt.close() 
        cv2.imwrite(os.path.join(output_dir, f"sam_mask_{img_file}.png"), sam_mask * 255) 
    return { 
        'dice': dice_score, 
        'iou': iou, 
        'hd': hd, 
        'hd95': hd95 
