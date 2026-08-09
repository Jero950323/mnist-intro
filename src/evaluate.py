"""评估与可视化：准确率、混淆矩阵、错误样例、网络内部特征。

用法（在项目根目录）:
    python -m src.evaluate --model mlp
    python -m src.evaluate --model cnn

所有图表会保存到 outputs/ 目录，用于分享讲解。
"""

import argparse
import json
import os
from pathlib import Path

# 限制数值库线程数，避免低内存机器上 OpenBLAS 分配失败
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib

if not os.environ.get("IN_NOTEBOOK"):
    matplotlib.use("Agg")  # 命令行运行时不弹窗，直接存图
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

try:
    from .data import denormalize, get_loaders
    from .model import build_model
    from .preprocess import preprocess_digit
except ImportError:
    from data import denormalize, get_loaders
    from model import build_model
    from preprocess import preprocess_digit

# 中文字体配置（Windows 自带微软雅黑/黑体）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_accuracy(model, loader, device):
    """返回测试集准确率。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def plot_history(history, out_dir, model_name, show=False):
    """训练过程曲线：损失下降 + 准确率上升。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], "o-", label="训练损失")
    axes[0].set_title("损失曲线（越小越好）")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="训练准确率")
    axes[1].plot(epochs, history["test_acc"], "s-", label="测试准确率")
    axes[1].set_title("准确率曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"{model_name}_history.png"
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(model, loader, device, out_dir, model_name, show=False):
    """混淆矩阵：看清哪些数字之间容易被认混。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in range(10)])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name.upper()} 混淆矩阵（测试准确率 {acc:.2%}）")
    fig.tight_layout()

    path = out_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig, acc


def plot_error_examples(model, loader, device, out_dir, model_name, n=15, show=False):
    """把模型认错的样例挑出来，分析为什么错。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    errors = []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for img, lab, pred in zip(images, labels, preds):
                if lab.item() != pred.item():
                    errors.append((img, lab.item(), pred.item()))
                    if len(errors) >= n:
                        break
            if len(errors) >= n:
                break

    cols = 5
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axes = np.atleast_1d(axes).ravel()
    for ax, (img, lab, pred) in zip(axes, errors):
        ax.imshow(denormalize(img).squeeze(), cmap="gray")
        ax.set_title(f"真:{lab} 测:{pred}", color="red", fontsize=10)
        ax.axis("off")
    for ax in axes[len(errors) :]:
        ax.axis("off")
    fig.suptitle("模型认错的样例（真=正确答案，测=模型预测）", fontsize=13)
    fig.tight_layout()

    path = out_dir / f"{model_name}_errors.png"
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_mlp_weights(model, out_dir, model_name="mlp", show=False):
    """把 MLP 第一层的权重画成小图：每个神经元学到的“笔画模板”。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w = model.fc1.weight.detach().cpu().numpy()  # (128, 784)
    w = w.reshape(w.shape[0], 28, 28)
    n = w.shape[0]
    cols = 16
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    vmax = max(abs(w.min()), abs(w.max()))
    for ax, k in zip(np.atleast_1d(axes).ravel(), range(n)):
        ax.imshow(w[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axis("off")
    fig.suptitle(
        "MLP 第一层 128 个神经元学到的模板（红=鼓励激活，蓝=抑制）", fontsize=13
    )
    fig.tight_layout()

    path = out_dir / f"{model_name}_weights.png"
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_cnn_filters_and_features(model, loader, device, out_dir, model_name="cnn", show=False):
    """展示 CNN 卷积核，以及网络处理输入时各层的特征图。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 第一层 32 个 3x3 卷积核
    filters = model.conv1.weight.detach().cpu().numpy()  # (32, 1, 3, 3)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for ax, k in zip(axes.ravel(), range(filters.shape[0])):
        ax.imshow(filters[k, 0], cmap="gray")
        ax.axis("off")
    fig.suptitle("CNN 第一层 32 个 3x3 卷积核（自动学到的边缘/笔画模板）", fontsize=13)
    fig.tight_layout()
    path = out_dir / f"{model_name}_filters.png"
    fig.savefig(path, dpi=150)
    if show:
        plt.show()

    # 2) 取一张测试图，看看各层对它产生了哪些特征图
    model.eval()
    images, _ = next(iter(loader))
    image = images[0:1].to(device)

    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()

        return hook

    handles = [
        model.conv1.register_forward_hook(make_hook("conv1")),
        model.conv2.register_forward_hook(make_hook("conv2")),
    ]
    with torch.no_grad():
        model(image)
    for h in handles:
        h.remove()

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    axes[0, 0].imshow(denormalize(image[0]).squeeze(), cmap="gray")
    axes[0, 0].set_title("输入", fontsize=10)
    axes[0, 0].axis("off")
    maps = activations["conv1"][0]  # (32, 28, 28)
    for ax, k in zip(axes.ravel()[1:], range(maps.shape[0])):
        ax.imshow(maps[k], cmap="hot")
        ax.axis("off")
    fig.suptitle(
        "CNN 第一层特征图：每个小图是网络在输入上“找”到的一种模式（越亮响应越强）",
        fontsize=13,
    )
    fig.tight_layout()

    path2 = out_dir / f"{model_name}_features.png"
    fig.savefig(path2, dpi=150)
    if show:
        plt.show()
    return fig


def save_example_digits(model, test_loader, out_dir, device=None, min_conf=0.99):
    """为 Gradio 演示挑选 0~9 各一张示例图（白底黑字）。

    不按“测试集里的第一张”选，而是挑模型**高置信度且识别正确**的样本，
    避免示例里出现容易认错的疑难写法（比如某些带横杠的 3 长得像 5）。
    """
    out_dir = Path(out_dir) / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chosen = {}
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            conf, preds = probs.max(dim=1)
            for img, lab, pred, c in zip(images, labels, preds, conf):
                lab = lab.item()
                if lab in chosen:
                    continue
                if pred.item() == lab and c.item() >= min_conf:
                    canvas = 1.0 - denormalize(img).squeeze().numpy()
                    plt.imsave(out_dir / f"digit_{lab}.png", canvas, cmap="gray")
                    chosen[lab] = round(float(c.item()), 4)
            if len(chosen) == 10:
                break
    return chosen


def save_challenge_examples(
    model, test_loader, out_dir, device=None, per_digit=2, max_total=12, min_conf=0.5
):
    """挑选模型会认错的疑难样本作为“挑战题”，演示时讲解模型的盲区。

    每个数字最多取 per_digit 张模型高置信度判错的图；
    元数据（正确答案 / 模型的判断 / 置信度）存到 examples/challenges.json。
    """
    out_dir = Path(out_dir) / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 先清掉旧挑战题图片，避免重新生成后残留过期文件
    for old in out_dir.glob("challenge_*.png"):
        old.unlink()

    errors = {d: [] for d in range(10)}
    with torch.no_grad():
        for images, labels in test_loader:
            for img, lab in zip(images, labels):
                lab = lab.item()
                # 关键：用和演示页完全相同的预处理路径再判一次，
                # 只保留“页面上确实会认错”的样本（有些原图判错的样本，
                # 经过裁剪/居中/对比度增强后反而会认对）
                res = preprocess_digit(denormalize(img).squeeze().numpy() * 255.0)
                if res is None:
                    continue
                tensor, _ = res
                probs = torch.softmax(model(tensor.to(device)), dim=1)[0]
                conf, pred = probs.max(dim=0)
                pred = int(pred.item())
                if pred != lab and float(conf.item()) >= min_conf:
                    errors[lab].append((float(conf.item()), img, pred))

    meta = {}
    for d in range(10):
        # 模型错得越“自信”越有教学效果，按置信度从高到低排
        errors[d].sort(key=lambda x: -x[0])
        for conf, img, pred in errors[d][:per_digit]:
            if len(meta) >= max_total:
                break
            name = f"challenge_{len(meta)}_{d}to{pred}.png"
            canvas = 1.0 - denormalize(img).squeeze().numpy()
            plt.imsave(out_dir / name, canvas, cmap="gray")
            meta[name] = {"true": d, "pred": pred, "conf": round(conf, 4)}
        if len(meta) >= max_total:
            break

    with open(out_dir / "challenges.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser(description="评估并可视化已训练的模型")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders(args.data_dir, batch_size=64)
    model = build_model(args.model)

    ckpt_path = Path(args.output_dir) / f"{args.model}_mnist.pth"
    if not ckpt_path.exists():
        print(f"未找到模型文件 {ckpt_path}，请先运行: python -m src.train --model {args.model}")
        return
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    history_path = Path(args.output_dir) / f"{args.model}_history.json"
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_history(history, args.output_dir, args.model)

    _, acc = plot_confusion_matrix(model, test_loader, device, args.output_dir, args.model)
    plot_error_examples(model, test_loader, device, args.output_dir, args.model)
    if args.model == "mlp":
        plot_mlp_weights(model, args.output_dir, args.model)
    else:
        plot_cnn_filters_and_features(model, test_loader, device, args.output_dir, args.model)

    chosen = save_example_digits(model, test_loader, args.output_dir, device=device)
    print(f"示例图已更新（0~9 各一张，模型置信度均 ≥99%）: {chosen}")
    challenges = save_challenge_examples(model, test_loader, args.output_dir, device=device)
    print(f"挑战题已生成（{len(challenges)} 张，均为模型认错的疑难样本）")
    print(f"评估完成：测试准确率 {acc:.2%}，所有图表已保存到 {args.output_dir}/")


if __name__ == "__main__":
    main()
