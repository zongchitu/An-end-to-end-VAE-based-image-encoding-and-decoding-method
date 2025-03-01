from .unet_parts import *
import torch.nn as nn
from .SE_Block import SEBlock
from .VS_Block import VariationalSampling


# UNet主网络
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, latent_dim=128, se=True, vs=True):
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

        self.se = se
        self.vs = vs

    def forward(self, x):
        if self.se:
            # 编码器
            x1 = self.inc(x)
            x1 = self.se1(x1)
            x2 = self.down1(x1)
            x2 = self.se2(x2)
            x3 = self.down2(x2)
            x3 = self.se3(x3)
            x4 = self.down3(x3)
            x4 = self.se4(x4)
            x5 = self.down4(x4)
            x5 = self.se5(x5)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

        if self.vs:
            # 变分采样
            z, mu, logvar = self.variational(x5)
            # 解码器
            x = self.up1(z, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits, mu, logvar
        else:
            # 解码器
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
            return logits
