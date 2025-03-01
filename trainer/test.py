import numpy as np
import torch
from torch import nn, device
from torch.utils.data import DataLoader
from evaluation.PSNR import calculate_psnr
from evaluation.SSIM import calculate_ssim
from data import denormalize


def test(
    model: nn.Module,
    weight_path: str,
    loader: DataLoader,
    device: device,
    vs: bool
):
    metric = {
        "psnr": [],
        "ssim": [],
    }
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if vs:
                logits, mu, logvar = model(x)
            else:
                logits = model(x)
            x, logits = denormalize(x).to(torch.int16), denormalize(logits).to(
                torch.int16
            )
            for origin_image, output_image in zip(x, logits):
                # 计算PSNR
                psnr = calculate_psnr(
                    origin_image.cpu().numpy().transpose(1, 2, 0),
                    output_image.cpu().numpy().transpose(1, 2, 0),
                )
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
