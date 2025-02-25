import torch
import torch.nn as nn
import torch.nn.functional as F


# 通道注意力模块，自适应调整通道权重
class SEBlock(nn.Module):
    
    # channel — 通道数；reduction - 压缩比
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 平均池化，tensor 的 shape变化：[B, C, H, W] → [B, C, 1, 1]，获得各通道的特征图
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 用于学习通道注意力权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
