import os
import torch
import pandas as pd
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from torch.amp import autocast
from models.sam_adapter import EnhancedSAM
from models.prompt_generator import HPSPGen
from datasets.echonet import EchoNetDataset
from utils.visualization import process_and_visualize
from utils.ef_utils import calculate_volume_from_mask, calculate_s_old, visualize_volume_geometry
from sklearn.metrics import r2_score, mean_squared_error

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
    
test_image_dir = cfg["data"]["image_dir"]
test_mask_dir = cfg["data"]["mask_dir"]
filelist_csv_path = cfg["data"]["filelist_csv"]
volumetracings_csv_path = cfg["data"]["volumetracings_csv"]
best_model_path = "checkpoints/best_lora_mspad_autoprompt.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    enhanced_sam = EnhancedSAM(
        model_type=cfg["model"]["type"],
        checkpoint=cfg["model"]["checkpoint"],
        lora_r=cfg["model"]["lora_r"]
    ) 
    model = HPSPGen(enhanced_sam.sam).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    model.eval()

    test_dataset = EchoNetDataset(test_image_dir, test_mask_dir)
    
    filelist_df = pd.read_csv(filelist_csv_path)
    filelist_df = filelist_df[filelist_df['Split'] == 'TEST']
    volume_tracings_df = pd.read_csv(volumetracings_csv_path)
    volume_tracings_df['FileName'] = volume_tracings_df['FileName'].str.replace('.avi', '').str.strip()
    print("FileList.csv head:", filelist_df.head())
    print("VolumeTracings.csv head:", volume_tracings_df.head())
 
    
    dice_dict = {}
 
    
    masks_dict = {}
    metrics = {'dice': [], 'iou': [], 'hd': [], 'hd95': []}
    inference_times = []
 
    for idx in range(len(test_dataset)):
        img_file = test_dataset.image_files[idx]
        mask_file = test_dataset.mask_files[idx]
        parts = img_file.split('_')
        filename = parts[0]
        phase = parts[1]
        frame_num = parts[2].split('.')[0]
        image_path = os.path.join(test_image_dir, img_file)
        mask_path = os.path.join(test_mask_dir, mask_file)
        try:
            start_time = time.perf_counter()
            visualize = (idx < 10)
            sam_mask, result = process_and_visualize(
                image_tensor=None,  
                mask_tensor=None,
                output_dir=output_dir,
                img_file=img_file,
                prompt_generator=model, 
                sam_predictor=enhanced_sam.sam,
                visualize=visualize,
                device=device,
                phase=phase
            )
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            for key in metrics:
                metrics[key].append(result[key])

            if filename not in dice_dict:
                dice_dict[filename] = {}
            dice_dict[filename][phase] = result['dice']
            if filename not in masks_dict:
                masks_dict[filename] = {}
            masks_dict[filename][phase] = sam_mask
        except Exception as e:
            print(f"Error processing image {idx+1}: {str(e)}")
 
    if metrics['dice']:
        print("\nAverage Metrics on Test Set (All Samples, Auto Prompt):")
        print(f"Dice Score: {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
        print(f"IoU: {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
        print(f"Hausdorff Distance: {np.mean(metrics['hd']):.4f} ± {np.std(metrics['hd']):.4f}")
        print(f"95% Hausdorff Distance: {np.mean(metrics['hd95']):.4f} ± {np.std(metrics['hd95']):.4f}")
        print(f"Average Inference Time per Sample: {np.mean(inference_times):.4f} seconds")
        
        plt.figure(figsize=(12, 4))
        plt.hist(metrics['dice'], bins=int(1 / 0.01), color='#505050', range=(-0.05, 1.05))
        plt.xlim(-0.05, 1.05)
        plt.xticks([0, 0.25, 0.5, 0.75, 1.00])
        plt.xlabel('Overall DSC')
        plt.ylabel('Count')
        #plt.title('Overall DSC Histogram')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(os.path.join(output_dir, 'overall_dsc_hist.png'))
        plt.close()
        
        ed_dices = [dice_dict[filename].get('ED', 0.0) for filename in masks_dict if 'ED' in dice_dict[filename]]
        es_dices = [dice_dict[filename].get('ES', 0.0) for filename in masks_dict if 'ES' in dice_dict[filename]]
        
        if ed_dices:
            plt.figure(figsize=(12, 4))
            plt.hist(ed_dices, bins=int(1 / 0.01), color='#505050', range=(-0.05, 1.05))
            plt.xlim(-0.05, 1.05)
            plt.xticks([0, 0.25, 0.5, 0.75, 1.00])
            plt.xlabel('ED DSC')
            plt.ylabel('Count')
            #plt.title('ED DSC Histogram')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.savefig(os.path.join(output_dir, 'ed_dsc_hist.png'))
            plt.close()
        
        if es_dices:
            plt.figure(figsize=(12, 4))
            plt.hist(es_dices, bins=int(1 / 0.01), color='#505050', range=(-0.05, 1.05))
            plt.xlim(-0.05, 1.05)  # 明确设置x轴显示范围
            plt.xticks([0, 0.25, 0.5, 0.75, 1.00])
            plt.xlabel('ES DSC')
            plt.ylabel('Count')
            #plt.title('ES DSC Histogram')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.savefig(os.path.join(output_dir, 'es_dsc_hist.png'))
            plt.close()
 
        
    ef_results = []
    ed_vis_count = 0
    es_vis_count = 0
    for filename in masks_dict:
        if 'ED' not in masks_dict[filename] or 'ES' not in masks_dict[filename]:
            print(f"Skipping {filename}: missing ED or ES mask")
            continue
        ed_mask = masks_dict[filename]['ED']
        es_mask = masks_dict[filename]['ES']
        s_old_ed, s_old_es = calculate_s_old(filename, volume_tracings_df, filelist_df) # 修改：返回 s_old_ed 和 s_old_es
        if s_old_ed is None or s_old_es is None:
            print(f"Skipping {filename}: failed to calculate s_old_ed or s_old_es")
            continue
        edv_model, ed_geometry = calculate_volume_from_mask(ed_mask, visualize=(ed_vis_count < 20), spacing=s_old_ed / 2) # 修改：除以2
        esv_model, es_geometry = calculate_volume_from_mask(es_mask, visualize=(es_vis_count < 20), spacing=s_old_es / 2) # 修改：除以2
        if edv_model == 0:
            ef_model = 0.0
        else:
            ef_model = (edv_model - esv_model) / edv_model * 100
        original_row = filelist_df[filelist_df['FileName'] == filename]
        if original_row.empty:
            print(f"Skipping {filename}: not found in FileList.csv")
            continue
        ef_original = original_row['EF'].values[0]
        edv_original = original_row['EDV'].values[0]
        esv_original = original_row['ESV'].values[0]
        # 获取对应的 Dice 分数从 dice_dict
        ed_dice = dice_dict[filename].get('ED', 0.0)
        es_dice = dice_dict[filename].get('ES', 0.0)
        ef_abs_error = abs(ef_model - ef_original)
        edv_abs_error = abs(edv_model - edv_original)
        esv_abs_error = abs(esv_model - esv_original)
        mape_ef = ef_abs_error / ef_original * 100 if ef_original != 0 else 0
        ef_results.append({
            'FileName': filename,
            'EF_model': ef_model,
            'EF_original': ef_original,
            'EF_abs_error': ef_abs_error,
            'EDV_Dice': ed_dice, # 添加 EDV_Dice
            'EDV_model': edv_model,
            'EDV_original': edv_original,
            'EDV_abs_error': edv_abs_error,
            'ESV_Dice': es_dice, # 添加 ESV_Dice
            'ESV_model': esv_model,
            'ESV_original': esv_original,
            'ESV_abs_error': esv_abs_error,
            'MAPE': mape_ef, # 恢复 MAPE 键
        })
        if ed_geometry and ed_vis_count < 10:
            visualize_volume_geometry(ed_mask, ed_geometry, filename, 'ED', volume_vis_dir)
            ed_vis_count += 1
        if es_geometry and es_vis_count < 10:
            visualize_volume_geometry(es_mask, es_geometry, filename, 'ES', volume_vis_dir)
            es_vis_count += 1
 
    if ef_results:
        ef_df = pd.DataFrame(ef_results)
        print("\nEF Evaluation Results:")
        print(ef_df)
        mae_ef = np.mean(ef_df['EF_abs_error'])
        mape_ef = np.mean(ef_df['MAPE'])
        std_ef = np.std(ef_df['EF_abs_error'])
        print("\nEF Metrics:")
        print(f"Average Absolute Error (MAE): {mae_ef:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_ef:.4f}%")
        print(f"Standard Deviation of Absolute Error (STD): {std_ef:.4f}")
        abs_error_edv = ef_df['EDV_abs_error']
        mae_edv = np.mean(abs_error_edv)
        mape_edv = np.mean(abs_error_edv / ef_df['EDV_original'] * 100) if not ef_df['EDV_original'].eq(0).any() else 0
        std_edv = np.std(abs_error_edv)
        print("\nEDV Metrics:")
        print(f"Average Absolute Error (MAE): {mae_edv:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_edv:.4f}%")
        print(f"Standard Deviation of Absolute Error (STD): {std_edv:.4f}")
        abs_error_esv = ef_df['ESV_abs_error']
        mae_esv = np.mean(abs_error_esv)
        mape_esv = np.mean(abs_error_esv / ef_df['ESV_original'] * 100) if not ef_df['ESV_original'].eq(0).any() else 0
        std_esv = np.std(abs_error_esv)
        print("\nESV Metrics:")
        print(f"Average Absolute Error (MAE): {mae_esv:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape_esv:.4f}%")
        print(f"Standard Deviation of Absolute Error (STD): {std_esv:.4f}")
        ef_df.to_csv(os.path.join(output_dir, "ef_evaluation.csv"), index=False)
     
        # EF scatter plot 
        plt.figure(figsize=(8, 6))
        plt.scatter(ef_df['EF_model'], ef_df['EF_original'], alpha=0.7)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Model EF (%)')
        plt.ylabel('Original EF (%)')
        plt.title('Scatter Plot: Model EF vs Original EF')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ef_scatter_plot.png'))
        plt.close()
        r2_ef = r2_score(ef_df['EF_original'], ef_df['EF_model'])
        rmse_ef = np.sqrt(mean_squared_error(ef_df['EF_original'], ef_df['EF_model']))
        print(f"R² for EF: {r2_ef:.4f}")
        print(f"RMSE for EF: {rmse_ef:.4f}")
     
        # Bland-Altman plot for EF
        mean_ef = (ef_df['EF_original'] + ef_df['EF_model']) / 2
        diff_ef = ef_df['EF_model'] - ef_df['EF_original']
        plt.figure(figsize=(8, 6))
        plt.scatter(mean_ef, diff_ef, alpha=0.7)
        plt.axhline(np.mean(diff_ef), color='r', linestyle='--', label='Mean Difference')
        plt.axhline(np.mean(diff_ef) + 1.96 * np.std(diff_ef), color='g', linestyle='--', label='+1.96 SD')
        plt.axhline(np.mean(diff_ef) - 1.96 * np.std(diff_ef), color='g', linestyle='--', label='-1.96 SD')
        plt.xlabel('Mean EF (%)')
        plt.ylabel('Difference (Model - Original) (%)')
        plt.title('Bland-Altman Plot for EF')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'ef_bland_altman.png'))
        plt.close()
     
        # EDV scatter plot (互换轴)
        plt.figure(figsize=(8, 6))
        plt.scatter(ef_df['EDV_model'], ef_df['EDV_original'], alpha=0.7)
        plt.plot([ef_df['EDV_model'].min(), ef_df['EDV_model'].max()], [ef_df['EDV_model'].min(), ef_df['EDV_model'].max()], 'r--')
        plt.xlabel('Model EDV (ml)')
        plt.ylabel('Original EDV (ml)')
        plt.title('Scatter Plot: Model EDV vs Original EDV')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'edv_scatter_plot.png'))
        plt.close()
        r2_edv = r2_score(ef_df['EDV_original'], ef_df['EDV_model'])
        rmse_edv = np.sqrt(mean_squared_error(ef_df['EDV_original'], ef_df['EDV_model']))
        print(f"R² for EDV: {r2_edv:.4f}")
        print(f"RMSE for EDV: {rmse_edv:.4f}")
     
        # Bland-Altman plot for EDV (不改)
        mean_edv = (ef_df['EDV_original'] + ef_df['EDV_model']) / 2
        diff_edv = ef_df['EDV_model'] - ef_df['EDV_original']
        plt.figure(figsize=(8, 6))
        plt.scatter(mean_edv, diff_edv, alpha=0.7)
        plt.axhline(np.mean(diff_edv), color='r', linestyle='--', label='Mean Difference')
        plt.axhline(np.mean(diff_edv) + 1.96 * np.std(diff_edv), color='g', linestyle='--', label='+1.96 SD')
        plt.axhline(np.mean(diff_edv) - 1.96 * np.std(diff_edv), color='g', linestyle='--', label='-1.96 SD')
        plt.xlabel('Mean EDV (ml)')
        plt.ylabel('Difference (Model - Original) (ml)')
        plt.title('Bland-Altman Plot for EDV')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'edv_bland_altman.png'))
        plt.close()
     
        # ESV scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(ef_df['ESV_model'], ef_df['ESV_original'], alpha=0.7)
        plt.plot([ef_df['ESV_model'].min(), ef_df['ESV_model'].max()], [ef_df['ESV_model'].min(), ef_df['ESV_model'].max()], 'r--')
        plt.xlabel('Model ESV (ml)')
        plt.ylabel('Original ESV (ml)')
        plt.title('Scatter Plot: Model ESV vs Original ESV')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'esv_scatter_plot.png'))
        plt.close()
        r2_esv = r2_score(ef_df['ESV_original'], ef_df['ESV_model'])
        rmse_esv = np.sqrt(mean_squared_error(ef_df['ESV_original'], ef_df['ESV_model']))
        print(f"R² for ESV: {r2_esv:.4f}")
        print(f"RMSE for ESV: {rmse_esv:.4f}")
     
        # Bland-Altman plot for ESV
        mean_esv = (ef_df['ESV_original'] + ef_df['ESV_model']) / 2
        diff_esv = ef_df['ESV_model'] - ef_df['ESV_original']
        plt.figure(figsize=(8, 6))
        plt.scatter(mean_esv, diff_esv, alpha=0.7)
        plt.axhline(np.mean(diff_esv), color='r', linestyle='--', label='Mean Difference')
        plt.axhline(np.mean(diff_esv) + 1.96 * np.std(diff_esv), color='g', linestyle='--', label='+1.96 SD')
        plt.axhline(np.mean(diff_esv) - 1.96 * np.std(diff_esv), color='g', linestyle='--', label='-1.96 SD')
        plt.xlabel('Mean ESV (ml)')
        plt.ylabel('Difference (Model - Original) (ml)')
        plt.title('Bland-Altman Plot for ESV')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'esv_bland_altman.png'))
        plt.close()
    else:
        print("No EF results calculated.")
