import os

import cv2
import numpy as np


def calculate_psnr(original_image, output_image):
    # 对于单张图片，计算PSNR
    # original_image : 原始图像，即gt
    # output_image : 经过VAE编码解码两个过程最终输出的图像

    # 确保两张图像的尺寸相同
    if original_image.shape != output_image.shape:
        raise ValueError("The dimensions of the images must be the same")

    # 转换为浮动类型，防止溢出问题
    original_image = original_image.astype(np.float32)
    output_image = output_image.astype(np.float32)

    # 计算均方误差 (MSE)
    mse = np.mean((original_image - output_image) ** 2)

    # 如果均方误差为零，表示两图像完全相同,提前返回inf，防止后面除零错误
    if mse < 1e-7:
        return float('inf')

    # 计算最大像素值，通常对于8位图像而言是255
    max_pixel = 255.0

    # 计算PSNR
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


if __name__ == "__main__":
    # 读取图像文件
    original_image = cv2.imread(os.path.join("example", "bees_1.jpg"))  # 原始图像
    output_image = cv2.imread(os.path.join("example", "bees_2.jpg"))  # 压缩图像

    # 计算PSNR
    psnr_value = calculate_psnr(original_image, output_image)

    print(f"PSNR value: {psnr_value:.3f} dB")
