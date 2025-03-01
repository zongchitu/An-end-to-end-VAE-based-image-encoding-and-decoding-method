from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from data import get_cifar10_dataloader
from unet.unet_model import UNet
from trainer.train import train
from trainer.test import test
from utils import seed_all
import argparse

arg = argparse.ArgumentParser()

arg.add_argument("--device", type=str, required=False, default='cpu')
arg.add_argument("--seed", type=int, required=False, default=42)
arg.add_argument("--n_channels", type=int, required=False, default=3)
arg.add_argument("--n_classes", type=int, required=False, default=3)
arg.add_argument("--lr", type=float, required=False, default=1e-4)
arg.add_argument("--step", type=int, required=False, default=100)
arg.add_argument("--pre_trained", type=bool, required=False, default=0)
arg.add_argument("--w/o_SE_block", type=bool, required=False, default=0)
arg.add_argument("--w/o_VS_block", type=bool, required=False, default=0)
params = arg.parse_args().__dict__


if __name__ == "__main__":

    # 固定种子
    seed_all(params["seed"])

    # 数据加载
    train_loader, val_loader, test_loader = get_cifar10_dataloader()

    # 指定设备
    device = params["device"]

    # 是否消融（是否加入“通道注意力模块” 及 “变分模块”）
    se = params["w/o_SE_block"]
    vs = params["w/o_VS_block"]

    # 模型建立
    model = UNet(n_channels=params["n_channels"], n_classes=params["n_classes"], bilinear=True, latent_dim=512, se=params["w/o_SE_block"], vs=params["w/o_VS_block"]).to(device)

    # 模式选择：训练 or 测试
    pre_trained = params["pre_trained"]

    if not pre_trained:
        optimizer = Adam(model.parameters(), lr=params["lr"])
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        train(model, 100, train_loader, optimizer, scheduler, device, vs)

    else:
        test(model, "./log/model_last.pth", test_loader, device, vs)
