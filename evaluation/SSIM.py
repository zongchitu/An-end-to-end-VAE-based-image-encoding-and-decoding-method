from skimage.metrics import structural_similarity as ssim


def calculate_ssim(original_image, output_image):
    # 对于单张图片，计算SSIM,这里是RGB/RGBA每个通道都计算SSIM值然后平均
    # original_image : 原始图像，即gt
    # output_image : 经过VAE编码解码两个过程最终输出的图像
    # 确保两张图像的尺寸相同
    if original_image.shape != output_image.shape:
        raise ValueError("The dimensions of the images must be the same")
    # 初始化SSIM值
    ssim_total = 0.0
    channels = original_image.shape[2]  # 通道数（通常为3）
    # 对每个通道分别计算SSIM并求平均
    for i in range(channels):
        original_channel = original_image[:, :, i]
        output_channel = output_image[:, :, i]
        # 计算SSIM
        ssim_value, _ = ssim(original_channel, output_channel, full=True)
        # ssim_value : 最终的整体SSIM评分，表示两幅图像（或两个通道）在结构上的相似性。
        # _ : 一个与输入图像大小相同的矩阵，其中每个像素值代表该位置的SSIM值，可以帮助你分析图像的局部结构相似度。
        ssim_total += ssim_value
    # 计算平均SSIM
    return ssim_total / channels
