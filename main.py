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

device = 0
writer = SummaryWriter(log_dir="log")
vgg_model = models.vgg16(pretrained=True).features.eval().to(device)


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


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_dataloader()
    model = UNet(n_channels=3, n_classes=3, bilinear=True, latent_dim=512).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    train(model, 100, train_loader, optimizer, scheduler)
