import torch.nn as nn
import torch.nn.functional as F
from models.sam_adapter import EnhancedSAM

class SAMFeatureExtractor(nn.Module):
    def __init__(self, sam_model, feature_layers=[7, 9, 10, 11]):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.feature_layers = feature_layers
    def forward(self, x):
        features = []
        # 手动运行image_encoder并收集指定层输出
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed
        for i, blk in enumerate(self.image_encoder.blocks):
            x = blk(x)
            if i in self.feature_layers:
                features.append(x)
        return features

# 定义ASPP模块（Atrous Spatial Pyramid Pooling）
class ASPP(nn.Module):
    def __init__(self, in_channels=512, atrous_rates=[6, 12], branch_channels=128, out_channels=512):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # 分支1: 1x1 conv
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 分支2-3: 只保留 rate=6 和 rate=12（删除rate=18）
        for rate in atrous_rates:
            self.branches.append(nn.Sequential(
                # Depthwise Atrous Conv
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # Pointwise Conv
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 分支5: 全局平均池化 + 1x1 conv
        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        ))
    
        # 融合层：现在是 4 × 128 = 512 通道 → 512（完美对齐）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(self.branches) * branch_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch_outputs = []
        # 前3个分支直接应用（1x1 + rate6 + rate12）
        for branch in self.branches[:3]:
            branch_outputs.append(branch(x))
        
        # 只剩最后一个分支需要上采样（全局平均池化）
        for branch in self.branches[3:]:
            global_out = branch(x)
            global_out = F.interpolate(global_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            branch_outputs.append(global_out)
        
        # 融合
        fused = torch.cat(branch_outputs, dim=1)  # [B, 512, H, W]
        return self.fusion_conv(fused)

# 2. 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        x = self.up(x)        # 改这里
        x = self.conv(x)
        x = self.cbam(x)
        return x

# 定义CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        # 通道注意力
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(concat))
        x = x * spatial_att
        return x

# 3. 基于ViT的自动提示生成器
class ViTPromptGenerator(nn.Module):
    def __init__(self, sam_model, num_classes=1):
        super().__init__()
        self.feature_extractor = SAMFeatureExtractor(sam_model)
        try:
            self.embed_dim = sam_model.image_encoder.embed_dim
        except AttributeError:
            if hasattr(sam_model.image_encoder, 'patch_embed') and hasattr(sam_model.image_encoder.patch_embed, 'proj'):
                self.embed_dim = sam_model.image_encoder.patch_embed.proj.out_channels
            else:
                self.embed_dim = 768
            print(f"Warning: 'embed_dim' not found, using fallback value: {self.embed_dim}")
        # 新增：每层特征单独投影到256通道（如果你用了4层就是1024通道，3层就是768通道）
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(len(self.feature_extractor.feature_layers))  # 自动匹配你上面设置的层数
        ])

        # 融合层：现在输入通道最多1024（4×256）
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * len(self.feature_extractor.feature_layers), 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)  # 可选：我建议加个Dropout2d，比普通Dropout更稳
        )
        self.aspp = ASPP(in_channels=512, atrous_rates=[6, 12], branch_channels=128, out_channels=512)  # 新增ASPP模块
        self.decoder = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),     # ← 加上这层！→1024x1024
        )
        # final_conv 输入通道改成32
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, 1)
        )
    def forward(self, x):
        with autocast(device_type='cuda'): # 更新为新API
            vit_features = self.feature_extractor(x)
            # 新方案：每层单独投影到256通道 → 再cat → 再融合
            projected = []
            for i, feat in enumerate(vit_features):                    # feat: [B, H, W, 768]
                x = feat.permute(0, 3, 1, 2)                           # → [B, 768, H, W]
                x = self.proj_layers[i](x)                             # → [B, 256, H, W]
                projected.append(x)

            fused = torch.cat(projected, dim=1)                        # → [B, 1024or768, 64, 64]
            fused = self.fusion(fused)                                 # → [B, 512, 64, 64]
            fused = self.aspp(fused)  # 应用ASPP
            x = self.decoder(fused)
            return self.final_conv(x) # 移除sigmoid，输出logits 
