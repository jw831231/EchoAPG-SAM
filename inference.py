import torch
import os
import numpy as np
import time
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.sam_adapter import EnhancedSAM
from models.prompt_generator import ViTPromptGenerator
from datasets.camus import CAMUSDataset
from utils.visualization import process_and_visualize, dice_coefficient, iou_score, hausdorff_distance, hausdorff_distance_95

if __name__ == "__main__":

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading EnhancedSAM + AutoPrompt model...")
    sam = EnhancedSAM(
        model_type=cfg["model"]["type"],
        checkpoint=cfg["model"]["checkpoint"],
        lora_r=cfg["model"]["lora_r"]
    ).to(device)

    model = ViTPromptGenerator(sam.sam).to(device)

    best_path = "checkpoints/best_lora_mspad_autoprompt.pth" 
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"✅ Loaded best model from {best_path}")
    else:
        print("⚠️ 未找到最佳模型，使用当前权重")

    model.eval()

    test_dataset = CAMUSDataset(
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=None,
        patient_list=None
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    metrics = {'dice': [], 'iou': [], 'hd': [], 'hd95': []}
    inference_times = []

    print(f"开始测试，共 {len(test_dataset)} 张图像...")

    for idx, (image, mask, img_file) in enumerate(test_loader):
        image = image.to(device)
        mask = mask.to(device).float()
        img_file = img_file[0]

        try:
            start_time = time.perf_counter()
            visualize = (idx < 30) 

            result = process_and_visualize(
                image_tensor=image,
                mask_tensor=mask,
                output_dir=output_dir,
                img_file=img_file,
                prompt_generator=model, 
                sam_predictor=sam.sam, 
                visualize=visualize,
                device=device
            )

            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            for k in metrics:
                metrics[k].append(result[k])

            print(f"[{idx+1}/{len(test_dataset)}] {img_file} | Dice: {result['dice']:.4f}")

        except Exception as e:
            print(f"Error on {img_file}: {e}")

    if metrics['dice']:
        print("\n" + "="*60)
        print("🎉 测试集平均指标（LoRA+MSPAd + 智能AutoPrompt）")
        print("="*60)
        print(f"Dice  : {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"IoU   : {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
        print(f"HD    : {np.mean([x for x in metrics['hd'] if x != float('inf')]):.2f}")
        print(f"HD95  : {np.mean([x for x in metrics['hd95'] if x != float('inf')]):.2f}")
        print(f"平均推理时间: {np.mean(inference_times):.3f} 秒/张")
        print("="*60)

        np.save(os.path.join(output_dir, "test_metrics.npy"), metrics)
