import torch
import torchvision.models as models
from torch import nn
from torch.utils.tensorboard import writer, SummaryWriter
from .loss_citeration import compute_total_loss, compute_total_loss_standard
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR


vgg_model = models.vgg16(pretrained=True).features.eval().to('cpu')


def train(
    model: nn.Module,
    max_step: int,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: StepLR,
    device: str,
    vs: bool
):
    writer = SummaryWriter(log_dir="log")
    for step, (x, y) in enumerate(loader):
        if step >= max_step:
            break
        # 训练代码
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if vs:
            logits, mu, logvar = model(x)
            loss = compute_total_loss(
                model_output=logits,
                model_input=x,
                vgg_model=vgg_model,
                mu=mu,
                logvar=logvar,
            )
        else:
            x_recon = model(x)
            loss = compute_total_loss_standard(
                model_output=x_recon,
                model_input=x,
                vgg_model=vgg_model
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
