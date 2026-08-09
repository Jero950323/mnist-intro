"""模型定义：两个经典网络，一个全连接（MLP），一个卷积（CNN）。"""

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """全连接网络（多层感知机）：784 -> 128 -> 10。

    思路：把 28x28 的图片拉平成 784 个数字，每个神经元对 784 个数字
    做加权求和，再经过激活函数。适合讲原理，但忽略了像素的空间关系。
    """

    def __init__(self, hidden=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 把 (batch, 1, 28, 28) 展平成 (batch, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """卷积网络：Conv -> Pool -> Conv -> Pool -> 全连接。

    思路：先用小模板（卷积核）在图片上滑动，自动学习“边缘、弧线”等
    局部特征，再逐层组合成更复杂的笔画，最后用全连接层做分类。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        self.pool = nn.MaxPool2d(2)  # 尺寸减半：28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(name="mlp"):
    """按名称创建模型，name 可选 'mlp' 或 'cnn'。"""
    name = name.lower()
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    raise ValueError(f"未知模型: {name}，请选择 mlp 或 cnn")


if __name__ == "__main__":
    for model_name in ["mlp", "cnn"]:
        model = build_model(model_name)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name.upper()} 参数量: {n_params:,}")
        print(model)
        print()
