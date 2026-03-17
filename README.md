# EchoAPG-SAM: Automatic prompt generator for SAM in two-dimensional echocardiography left ventricular segmentation and ejection fraction estimation

**论文贡献**：
- 首次将 **LoRA + MSPAd + HPSPGen** 适配器与 **SAM** 融合
- 实现全自动心脏超声左心室分割和射血分数评估（无需人工提示）
- CAMUS数据集(左心室分割)：**Dice 93.34**，**IoU 87.77**，**HD 7.78**（SOTA）
- EchoNet-Dynamic数据集(左心室分割)：**Dice 91.27**，**IoU 84.22**，**HD95 2.74**（SOTA）
- EchoNet-Dynamic数据集（左心室射血分数评估）：**MAE 6.46**，**RMSE 8.21**，**R2 0.55**（SOTA）

## 快速启动
```bash
pip install -r requirements.txt
python train.py --config config.yaml
