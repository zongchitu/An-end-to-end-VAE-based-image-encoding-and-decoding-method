import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# 感知损失函数
def perceptual_loss(x, y, vgg_model, layers=None):
    """
    计算两张图像之间的感知损失

    参数:
    - x: 第一张图像，形状为 (batch_size, channels, height, width)
    - y: 第二张图像，形状为 (batch_size, channels, height, width)
    - vgg_model: 用于提取特征的VGG模型
    - layers: 用于计算损失的特定层索引，默认选定几层

    返回:
    - 感知损失的标量值
    """
    if layers is None:
        layers = [4, 9, 16, 23, 30]  # 默认选用VGG中的中间到高层

    # 提取两张图像的特征
    x_features = extract_features(x, vgg_model, layers)
    y_features = extract_features(y, vgg_model, layers)

    # 计算感知损失（均方误差）
    loss = 0.0
    for x_feat, y_feat in zip(x_features, y_features):
        loss += F.mse_loss(x_feat, y_feat)

    return loss


def extract_features(x, vgg_model, layers):
    """
    从VGG网络中提取特定层的特征

    参数:
    - x: 输入图像，形状为 (batch_size, channels, height, width)
    - vgg_model: 用于提取特征的VGG模型
    - layers: 要提取的层的索引列表

    返回:
    - 提取的特征列表
    """
    features = []
    layer_idx = 0
    for layer in vgg_model:
        x = layer(x)
        if layer_idx in layers:
            features.append(x)
        layer_idx += 1
    return features


def mae_mse_loss(x, y):
    """
    计算 MAE 和 MSE 损失。

    Parameters:
    - x: output (Tensor)
    - y: input(gt) (Tensor)

    Returns:
    - MAE 损失 (Tensor)
    - MSE 损失 (Tensor)
    """
    mae_loss = nn.functional.l1_loss(x, y)
    mse_loss = nn.functional.mse_loss(x, y)
    return mae_loss + mse_loss


def TV_loss(x):
    """
    计算图像的全变分损失（Total Variation Loss）。

    参数:
    - x: output，形状为 (batch_size, channels, height, width)

    返回:
    - TV Loss: 全变分损失的标量
    """
    # 计算水平方向和垂直方向的梯度
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]  # 水平方向的差异
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]  # 垂直方向的差异

    # 处理边界：通过填充使得 dx 和 dy 的尺寸一致
    dx = torch.cat([dx, torch.zeros_like(dx[:, :, :, :1])], dim=3)  # 为 dx 添加边界填充
    dy = torch.cat([dy, torch.zeros_like(dy[:, :, :1, :])], dim=2)  # 为 dy 添加边界填充

    # 计算梯度的平方
    dx = torch.pow(dx, 2)
    dy = torch.pow(dy, 2)

    # 计算梯度的平方根
    grad = torch.sqrt(dx + dy)

    # 对所有像素点求和
    tv_loss = grad.mean()

    return tv_loss


def latent_space_kl_divergence_loss(mu, logvar):
    """
    计算 VAE 的 KL 散度损失。衡量潜在空间的分布与标准正态分布之间的差异。

    Parameters:
    - mu: 编码器输出的均值 (Tensor)
    - logvar: 编码器输出的对数方差 (Tensor)

    Returns:
    - KL 散度损失 (Tensor)
    """
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss


def compute_total_loss(model_output, model_input, mu, logvar, vgg_model, layers=None):
    # 经验参数，请根据实际情况调整
    # MAE,MSE,潜在空间KL散度损失是主要的损失函数，权重大,给1
    # 感知损失,TV损失是辅助的损失函数，权重小，给0.1,0.01（经验参数）

    # 去掉感知损失能够正常运行
    # total_loss=mae_mse_loss(model_output,model_input) + 0.1*perceptual_loss(model_output,model_input,vgg_model,layers) + latent_space_kl_divergence_loss(mu, logvar) + 0.01*TV_loss(model_output)
    total_loss = (
        mae_mse_loss(model_output, model_input)
        + latent_space_kl_divergence_loss(mu, logvar)
        + 0.01 * TV_loss(model_output)
    )
    return total_loss


if __name__ == "__main__":

    # 输入图像 (假设已经通过其他方式加载并预处理为张量)
    x = torch.rand((1, 3, 256, 256))  # 假设生成图像
    y = torch.rand((1, 3, 256, 256))  # 假设目标图像
    mu = torch.rand((1, 20))  # 假设编码器输出的均值
    logvar = torch.rand((1, 20))  # 假设编码器输出的对数方差

    # 加载VGG16模型并去掉分类层
    vgg16 = models.vgg16(pretrained=True).features.eval()

    # 计算损失
    loss = compute_total_loss(x, y, mu, logvar, vgg16)
    print(f"total Loss: {loss.item()}")
