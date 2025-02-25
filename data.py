import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader

# 配置参数
batch_size = 8
generator = torch.Generator().manual_seed(42)

# 图像预处理
transform = v2.Compose(
    [
        v2.ToTensor(),
        v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 的均值
            std=[0.247, 0.243, 0.261],  # CIFAR-10 的标准差
        ),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(90),
    ]
)


# 可能用到的反标准化函数
def denormalize(tensor):
    """
    反标准化图像
    :param tensor: 标准化后的图像张量
    :return: 反标准化后的图像张量
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 反标准化的公式：t = (t * s) + m
    return tensor


def get_cifar10_dataloader():
    """获取 CIFAR-10 数据集的数据加载器

    Returns:
        _type_: _description_
    """
    train_dataset = torchvision.datasets.CIFAR10(
        root="./datasets/cifar10", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./datasets/cifar10", train=False, download=True, transform=transform
    )

    # 从训练集中再分出验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset=train_dataset, lengths=[0.8, 0.2], generator=generator
    )

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 每次训练时打乱数据
        num_workers=2,  # 使用多线程加载数据
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不需要打乱数据
        num_workers=2,
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不需要打乱数据
        num_workers=2,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_dataloader()
    pass
