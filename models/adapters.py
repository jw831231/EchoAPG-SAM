import torch
import torch.nn as nn
import torch.nn.functional as F

class MSPAd(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ln_down = nn.LayerNorm(dim)
        
        self.branches = nn.ModuleList()
        for kernel_size in [1, 3, 5]:
            padding = (kernel_size - 1) // 2
            depthwise = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                padding=padding, groups=dim, bias=True)
            pointwise = nn.Conv2d(dim, dim // 3, kernel_size=1, bias=True)
            self.branches.append(nn.Sequential(depthwise, pointwise))
        
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2, bias=True)
        self.ln_out = nn.LayerNorm(dim)
        self.adapter_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, H, W]
        x_down = self.downsample(x)                    # [B, C, H/2, W/2]
        B, C, H_down, W_down = x_down.shape
        x_down = x_down.permute(0, 2, 3, 1)            # [B, H_down, W_down, C]
        x_down = self.ln_down(x_down)
        x_down = x_down.permute(0, 3, 1, 2)            # 回 [B, C, H_down, W_down]
        
        branch_outputs = [branch(x_down) for branch in self.branches]
        concat = torch.cat(branch_outputs, dim=1)      # [B, dim, H_down, W_down]
        fused = self.fuse(concat)
        
        up = self.upsample(fused)                      # [B, dim, H, W]
        up = up.permute(0, 2, 3, 1)                    # [B, H, W, dim]
        up = self.ln_out(up)
        up = up.permute(0, 3, 1, 2)                    # 回 [B, dim, H, W]
        
        return self.adapter_scale * up
