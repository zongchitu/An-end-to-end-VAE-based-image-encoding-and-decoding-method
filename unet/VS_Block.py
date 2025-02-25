import torch
import torch.nn as nn
import torch.nn.functional as F


# 变分采样，增大域视野
class VariationalSampling(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super(VariationalSampling, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 编码器输出均值和方差
        self.mu_layer = nn.Conv2d(in_channels, latent_dim, kernel_size=1)
        self.logvar_layer = nn.Conv2d(in_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        # 计算均值和方差       
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar
