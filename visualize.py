from unet.unet_model import UNet
from utils import seed_all
from data import get_cifar10_dataloader
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
from data import denormalize
import plotly.express as px
from PIL import Image
import numpy as np
import cv2


def visualize(weight_path: Path):
    _, _, test_loader = get_cifar10_dataloader()
    model = UNet(
        n_channels=3, n_classes=3, bilinear=True, latent_dim=512, se=True, vs=True
    )
    seed_all(42)
    model.load_state_dict(torch.load(weight_path))
    count = 0
    with torch.no_grad():
        for img, _ in test_loader:
            count += 1
            output = model(img)
            pred, _, _ = model(img)
            img_numpy = pred[0].cpu().numpy()
            img_numpy = np.transpose(img_numpy, (1, 2, 0))
            img_cv2 = img_numpy * 255
            img_cv2 = img_cv2.astype("uint8")
            img_bgr = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{count}_pred.png", img_cv2)
            cv2.imwrite(f"{count}_pred_rgbbgr.png", img_bgr)
            cv2.imwrite(
                f"{count}_img.png", img[0].cpu().numpy().transpose(1, 2, 0) * 255
            )
            # img = denormalize(img[0]).cpu().numpy().transpose(1, 2, 0)
            # pred = denormalize(pred[0]).cpu().numpy().transpose(1, 2, 0)
            # px.imshow(img * 255).write_image("img.png")
            # px.imshow(pred * 255).write_image("pred.png")
            if count % 10 == 0:
                break


if __name__ == "__main__":
    visualize("log/model_last.pth")
