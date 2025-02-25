import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

device = 0
writer = SummaryWriter(log_dir="log")


def train(
    model: nn.Module,
    max_step: int,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: StepLR,
    criterion: nn.Module,
):
    for step, (x, y) in enumerate(loader):
        if step >= max_step:
            break
        # 训练代码
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # 记录训练过程
        writer.add_scalar("loss", loss.item(), global_step=step)
        if step % 10 == 0:
            print(f"step: {step}, loss: {loss.item()}")
    writer.close()
    return model
