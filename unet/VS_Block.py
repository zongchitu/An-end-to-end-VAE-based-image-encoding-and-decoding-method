import torch
import torch.nn as nn
import torch.nn.functional as F


# 变分采样，增大域视野
class VariationalSampling(nn.Module):

    def __init__(self, in_channels, gaussian_dim):
        super(VariationalSampling, self).__init__()
        self.in_channels = in_channels
        self.gaussian_dim = gaussian_dim
        self.flat_dim = in_channels*2*2

        # 编码器输出均值和方差
        self.mu_layer = nn.Linear(self.flat_dim, self.gaussian_dim)
        self.logvar_layer = nn.Linear(self.flat_dim, self.gaussian_dim)

        # 解码层
        self.decoder=nn.Sequential(
            nn.Linear(gaussian_dim, self.in_channels),
            nn.Tanh(),
            nn.Linear(self.in_channels, self.flat_dim),
            nn.Sigmoid() #图片被转为为0，1的值了，故用此函数
        )

    def forward(self, x):
        # (N,Ci,2,2)转(N,Ci*2*2)
        x = torch.flatten(x, 1)

        # 计算均值和方差       
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 解码回(N,Ci,2,2)
        z = self.decoder(z)
        z = z.reshape(-1, self.in_channels, 2, 2)

        return z, mu, logvar
