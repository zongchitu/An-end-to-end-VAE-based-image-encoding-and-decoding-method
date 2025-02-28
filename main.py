import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from loss_citeration import compute_total_loss
from data import get_cifar10_dataloader
from unet.unet_model import UNet
import torchvision.models as models
import torch
from evaluation.CompressionRatio import calculate_compression_ratio
from evaluation.PSNR import calculate_psnr
from evaluation.SSIM import calculate_ssim
from data import denormalize
import random
import numpy as np
import os

device = 0
writer = SummaryWriter(log_dir="log")
vgg_model = models.vgg16(pretrained=True).features.eval().to(device)


def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    model: nn.Module,
    max_step: int,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: StepLR,
):
    for step, (x, y) in enumerate(loader):
        if step >= max_step:
            break
        # 训练代码
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, mu, logvar = model(x)
        loss = compute_total_loss(
            model_output=logits,
            model_input=x,
            vgg_model=vgg_model,
            mu=mu,
            logvar=logvar,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        # 记录训练过程
        writer.add_scalar("loss", loss.item(), global_step=step)
        if step % 10 == 0:
            print(f"step: {step}, loss: {loss.item()}")
            torch.save(model.state_dict(), "./log/model_last.pth")
    writer.close()
    return model


def test(
    model: nn.Module,
    weight_path: str,
    loader: DataLoader,
):
    metric = {
        "psnr": [],
        "ssim": [],
    }
    model.load_state_dict(torch.load(weight_path, map_location=f"cuda:{device}"))
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits, mu, logvar = model(x)
            x, logits = denormalize(x).to(torch.int16), denormalize(logits).to(
                torch.int16
            )
            for origin_image, output_image in zip(x, logits):
                # 计算PSNR
                psnr = calculate_psnr(
                    origin_image.cpu().numpy().transpose(1, 2, 0),
                    output_image.cpu().numpy().transpose(1, 2, 0),
                )
                # if psnr == float("inf"):
                #     continue
                metric["psnr"].append(psnr)
                # 计算SSIM
                ssim = calculate_ssim(
                    origin_image.cpu().numpy().transpose(1, 2, 0),
                    output_image.cpu().numpy().transpose(1, 2, 0),
                )
                metric["ssim"].append(ssim)

                print(f"step: {step}, psnr: {psnr}, ssim: {ssim}")

        metric = {k: np.mean(v).item() for k, v in metric.items()}
        print(metric)
    return metric


if __name__ == "__main__":
    seed_all(42)
    train_loader, val_loader, test_loader = get_cifar10_dataloader()
    model = UNet(n_channels=3, n_classes=3, bilinear=True, latent_dim=512).to(device)
    # optimizer = Adam(model.parameters(), lr=1e-4)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    # train(model, 100, train_loader, optimizer, scheduler)
    test(model, "./log/model_last.pth", test_loader)
