""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F


# UNet主网络
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, latent_dim=128):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.latent_dim = latent_dim

        # 编码器部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 变分采样模块
        self.variational = VariationalSampling(1024 // factor, latent_dim)

        # 解码器部分
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # SE模块
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        self.se5 = SEBlock(1024 // factor)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x1 = self.se1(x1)
        x2 = self.down1(x1)
        x2 = self.se2(x2)
        x3 = self.down2(x2)
        x3 = self.se3(x3)
        x4 = self.down3(x2)
        x4 = self.se4(x4)
        x5 = self.down4(x3)
        x5 = self.se5(x5)

        # 变分采样
        z, mu, logvar = self.variational(x5)

        # 解码器
        x = self.up1(z, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits, mu, logvar


# 测试代码
if __name__ == "__main__":
    unet = UNet(n_channels=3, n_classes=1, bilinear=True, latent_dim=128)
    x = torch.randn(1, 3, 32, 32)
    logits, mu, logvar = unet(x)
    print("Logits shape:", logits.size())
    print("Mu shape:", mu.size())
    print("Logvar shape:", logvar.size())

