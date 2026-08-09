"""数据模块：自动下载 MNIST 数据集并创建 DataLoader。

MNIST 是公开免费的手写数字数据集：
- 6 万张训练图 + 1 万张测试图
- 每张 28x28 灰度图，标注 0~9
torchvision 会在第一次运行时自动从网上下载。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(data_dir="data", batch_size=64, num_workers=0, download=True, augment=False):
    """返回 (train_loader, test_loader)。

    参数:
        data_dir: 数据集存放目录，第一次运行会自动下载
        batch_size: 每批图片数量（训练时一次看多少张）
        augment: 是否对训练数据做随机增强（旋转/平移/缩放）。
                 手写输入往往歪一点、偏一点，增强后的模型更稳。
    """
    data_dir = Path(data_dir)

    # 数据预处理：转成张量，并做标准化（让像素值均值约 0、方差约 1，训练更稳定）
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    else:
        train_transform = test_transform

    train_set = datasets.MNIST(
        data_dir, train=True, download=download, transform=train_transform
    )
    test_set = datasets.MNIST(
        data_dir, train=False, download=download, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def denormalize(tensor):
    """把标准化过的图像还原回 0~1 的灰度范围，方便画图展示。"""
    return tensor * 0.3081 + 0.1307


def main():
    train_loader, test_loader = get_loaders("data")
    images, labels = next(iter(train_loader))
    print(f"训练集: {len(train_loader.dataset)} 张, 测试集: {len(test_loader.dataset)} 张")
    print(f"一批数据的形状: {tuple(images.shape)}, 标签示例: {labels[:10].tolist()}")


if __name__ == "__main__":
    main()
