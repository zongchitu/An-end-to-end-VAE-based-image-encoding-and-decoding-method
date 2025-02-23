import os
import numpy as np
import cv2


def calculate_compression_ratio(original_image_path, latent_dim, dtype=np.float32):
    # 计算压缩比compression_ratio
    # 潜在空间向量 : 经过已经训练好的VAE编码器之后，解码器之前的向量，即latent_vector
    # original_image : 任意原始图像1张
    # latent_dim : 潜在空间向量的维度，即latent_vector展平为1维后的size
    # dtype : 潜在空间向量所存储的数的数据类型，如np.float32

    # 获取原始图像的文件大小（单位：字节）
    original_size = os.path.getsize(original_image_path)

    # 每个元素的字节大小，根据数据类型（dtype）
    element_size = np.dtype(dtype).itemsize  # 获取每个元素的字节大小

    # 计算潜在空间表示的大小：latent_dim
    latent_space_size = latent_dim * element_size

    # 计算压缩比
    compression_ratio = original_size / latent_space_size
    return compression_ratio


if __name__ == '__main__':
    # 读取图像文件
    original_image_path = os.path.join('example','bees_1.jpg')  # 原始图像
    latent_dim=32*32  # 潜在空间向量展平为1维后的size
    dtype=np.float32  # 潜在空间向量所存储的数的数据类型

    # 计算PSNR
    compression_ratio = calculate_compression_ratio(original_image_path, latent_dim,dtype)

    print(f"compression_ratio value: {compression_ratio:.3f}")

