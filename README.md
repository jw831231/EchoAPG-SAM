# EchoAPG-SAM: Automatic prompt generator for SAM in two-dimensional echocardiography left ventricular segmentation and ejection fraction estimation

**EchoAPG-SAM** is a novel framework that combines **LoRA + MSPAd adapters** with an **Intelligent Auto-Prompt Generator (HPSPGen)** to enable fully automatic left ventricular segmentation and ejection fraction estimation in cardiac ultrasound images using Segment Anything Model (SAM).

## Highlights
- First integration of **LoRA + MSPAd + HPSPGen** with SAM for medical imaging.
- **Zero manual prompts** — fully automatic (auto-generated box + point + mask_input).
- State-of-the-art performance on two public datasets:
  - **CAMUS** (LV segmentation): Dice **93.34%**, IoU **87.77%**, HD **7.78** (SOTA).
  - **EchoNet-Dynamic** (LV segmentation): Dice **91.27%**, IoU **84.22%**, HD95 **2.74** (SOTA).
  - **EchoNet-Dynamic** (EF estimation): MAE **6.46**, RMSE **8.21**, R² **0.55** (SOTA).

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jw831231/EchoAPG-SAM.git
cd EchoAPG-SAM

# 2. Install PyTorch and dependencies
# Install PyTorch with CUDA support (CUDA 12.1 recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Download the official SAM weights (Required)
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b.pth

# 4. Configure paths (Very Important!)
The default config.yaml uses Kaggle paths. You must modify it for local use:
# Create a local config copy
cp config.yaml config_local.yaml

# 5. Train the model
python train.py

# 6. Run inference & visualization
# Left ventricle segmentation + visualization (recommended first)
python inference.py

# Ejection Fraction (EF) estimation
python inference_ef.py

EchoAPG-SAM/
├── train.py                    # Main training script
├── inference.py                # Segmentation inference & visualization
├── inference_ef.py             # Ejection Fraction (EF) estimation
├── config.yaml                 # Configuration file
├── models/                     # EnhancedSAM (LoRA + MSPAd) + ViTPromptGenerator（HPSPGen）
├── datasets/                   # CAMUS & EchoNet-Dynamic loaders
│   ├── camus.py
│   └── echonet.py
├── utils/                      # Metrics, visualization & EF tools
│   ├── ef_utils.py
│   └── visualization.py
├── outputs/                    # Loss curves & result images
├── checkpoints/                # Model weights (currently empty)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md

Results Comparison

Table 1
Comparison of structure segmentation between our EchoAPG-SAM and other state-of-the-art methods on the CAMUS-LV dataset.
a represents the values came from EchoEFNet, b represents the values came from SAMUS.
| Method          | Dice   | IoU    | HD     |
|-----------------|--------|--------|--------|
| UNet       | 92.20  | 85.90  | -      |
| UNet++     | 91.95  | 85.55  | -      |
| SwinUNet   | 91.72  | 84.82  | 12.80  |
| SETR       | 92.82  | 86.76  | 11.34  |
| TransFuse  | 93.30  | 87.55  | 10.07  |
| DeepLabv3+ | 90.90  | 83.80  | -      |
| EchoEFNet  | 93.05  | 87.30  | -      |
| MedSAM     | 87.52  | 78.22  | 15.28  |
| MSA        | 90.95  | 83.70  | 11.29  |
| SAMed      | 87.67  | 78.67  | 13.24  |
| MemSAM      | 93.31  | 87.61  | -      |
| **EchoAPG-SAM** | **93.34** | **87.77** | **7.78** |

Table 2
Comparison of LV segmentation between our EchoAPG-SAM and other state-of-the-art methods on the EchoNet-Dynamic dataset (HD uses HD95). 
a represents the values came from MemSAM.
| Method          | Dice   | IoU    | HD95   |
|-----------------|--------|--------|--------|
| UNet       | 91.36  | 83.27  | 4.98   |
| SwinUNet   | 87.79  | 80.14  | 6.61   |
| H2Former   | 90.21  | 82.46  | 5.12   |
| MedSAM     | 86.47  | 79.19  | 7.97   |
| MSA        | 87.91  | 78.34  | 6.67   |
| SAMed      | 86.35  | 78.96  | 7.12   |
| SAMUS      | 91.79  | 84.32  | 5.35   |
| MemSAM     | 92.78  | 85.89  | 4.75   |
| **EchoAPG-SAM** | **91.27** | **84.22** | **2.74** |

Table 3
Comparison of LVEF estimations on EchoNet-Dynamic dataset.
EchoNet-Dynamic1 refers to testing using fixed 32 frames of data, and EchoNet-Dynamic2 refers to testing using beat-to-beat.
| Method               | Type      | MAE (%) | RMSE (%) | R²    | R     |
|----------------------|-----------|---------|----------|-------|-------|
| Regression           | Regression| 6.77    | 8.70     | 0.48  | -     |
| M                    | Regression| 5.95    | 8.38     | 0.52  | -     |
| R3D                  | Regression| 7.63    | 9.75     | 0.37  | -     |
| MC3                  | Regression| 6.59    | 9.39     | 0.42  | -     |
| EchoNet-Dynamic1     | Regression| 7.35    | 9.53     | 0.40  | -     |
| EchoNet-Dynamic2     | Regression| 4.05    | 5.32     | 0.81  | -     |
| Depth-Map            | Simpson   | 6.55    | -        | 0.61  | -     |
| MAEF-Net             | Simpson   | 6.29    | 8.21     | 0.54  | -     |
| MU-Net               | Simpson   | 6.61    | 8.91     | -     | -     |
| EchoSAM              | Simpson   | 6.39    | 8.56     | -     | -     |
| **EchoAPG-SAM**      | **Simpson**| **6.46**| **8.21** | **0.55** | -     |


Citation
@misc{EchoAPG-SAM2026,
  author = {Wei Jiang：Department of Basic Medical Sciences, Chongqing Medical and Pharmaceutical College, 401331, Chongqing, China},
  title = {EchoAPG-SAM: Automatic prompt generator for SAM in two-dimensional echocardiography left ventricular segmentation and ejection fraction estimation},
  year = {2026},
  url = {https://github.com/jw831231/EchoAPG-SAM}
}
