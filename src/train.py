"""训练脚本。

用法（在项目根目录）:
    python -m src.train --model mlp --epochs 5
    python -m src.train --model cnn --epochs 5

训练 5 个 epoch 后，模型文件和训练曲线数据会保存到 outputs/ 目录。
"""

import argparse
import json
import os
import time
from pathlib import Path

# 限制数值库线程数，避免低内存机器上 OpenBLAS 分配失败
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .data import get_loaders
    from .model import build_model
except ImportError:  # 直接运行 python src/train.py 时的兜底
    from data import get_loaders
    from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device, log_every=100):
    """训练一个 epoch：前向传播 -> 算损失 -> 反向传播 -> 更新参数。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # 前向：模型对这批图片的预测
        loss = criterion(outputs, labels)  # 计算预测和正确答案的差距
        loss.backward()  # 反向传播：算出每个参数该往哪个方向调
        optimizer.step()  # 梯度下降：把参数往减少损失的方向微调

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if log_every and (i + 1) % log_every == 0:
            print(f"    batch {i + 1}/{len(loader)}  loss={loss.item():.4f}")
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    """在测试集上评估准确率（不更新参数）。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 默认开启数据增强；Notebook 里直接调用 train() 时也兼容
    augment = getattr(args, "augment", True)
    train_loader, test_loader = get_loaders(
        args.data_dir, args.batch_size, augment=augment
    )
    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"模型: {args.model.upper()}，参数量: {n_params:,}，"
        f"数据增强: {'开' if augment else '关'}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        te_acc = evaluate(model, test_loader, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(
            f"  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
            f"test_acc={te_acc:.4f}  ({time.time() - t0:.1f}s)"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.model}_mnist.pth"
    torch.save(model.state_dict(), ckpt_path)
    with open(out_dir / f"{args.model}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"模型已保存: {ckpt_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 MNIST 手写数字识别模型")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp", help="模型类型")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-augment", action="store_false", dest="augment",
        help="关闭数据增强（默认开启：随机旋转/平移/缩放）",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    train(args)
