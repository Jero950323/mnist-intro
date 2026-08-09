"""把画板/图片预处理成 MNIST 风格输入的公共模块。

演示页（app/demo.py）和挑战题生成（src/evaluate.py）共用这一份逻辑，
保证页面上看到的识别结果和生成挑战题时的判断完全一致。
"""

import numpy as np
import torch
from PIL import Image


def preprocess_digit(img):
    """把画板图像处理成接近 MNIST 训练样本的样子。

    返回 (tensor, canvas)：
    - tensor: 标准化后的 (1, 1, 28, 28) 张量，直接喂给模型
    - canvas: 0~1 的 28x28 灰度图，用于页面上展示“模型看到的样子”

    关键点：
    1. 裁出数字本体；
    2. 保持宽高比缩放到 20x20 以内——直接拉伸成正方形会把细长的
       "1" 拉变形（容易认成 5/8），也会改变 "6" 的形状；
    3. 对比度增强，让细笔画更清晰；
    4. 按笔画质心居中放入 28x28 画布。
    画板空白时返回 None。
    """
    img = np.asarray(img, dtype=np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    if img.ndim == 3:  # RGBA/RGB 多通道先转灰度
        img = np.asarray(
            Image.fromarray(img.astype("uint8")).convert("L"), dtype=np.float32
        )
    img = img / 255.0

    # 统一成黑底白字（MNIST 的样子）
    if img.mean() > 0.5:
        img = 1.0 - img

    mask = img > 0.25
    if mask.sum() < 30:  # 几乎没有笔画
        return None

    # 裁出数字所在区域，四周留一点边距
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = max(2, int(0.05 * max(img.shape)))
    x0, x1 = max(x0 - pad, 0), min(x1 + pad, img.shape[1])
    y0, y1 = max(y0 - pad, 0), min(y1 + pad, img.shape[0])
    crop = img[y0:y1, x0:x1]

    # 保持宽高比：长边缩放到 20，短边等比缩放（避免变形）
    h, w = crop.shape
    scale = min(20.0 / w, 20.0 / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = np.asarray(
        Image.fromarray((crop * 255).astype("uint8"), mode="L").resize(
            size, Image.LANCZOS
        ),
        dtype=np.float32,
    ) / 255.0

    # 对比度增强：细笔画更清晰，观感更接近 MNIST 的粗笔画
    small = np.clip((small - 0.25) / 0.75, 0.0, 1.0)

    canvas = np.zeros((28, 28), dtype=np.float32)
    sh, sw = small.shape
    y0c, x0c = (28 - sh) // 2, (28 - sw) // 2
    canvas[y0c : y0c + sh, x0c : x0c + sw] = small

    # 按笔画质心微调居中（MNIST 数字基本都居中）
    mass = canvas > 0.1
    if mass.sum() > 0:
        my, mx = np.nonzero(mass)
        shift_y = int(np.clip(round(13.5 - my.mean()), -3, 3))
        shift_x = int(np.clip(round(13.5 - mx.mean()), -3, 3))
        if shift_y or shift_x:
            canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))

    # 和训练时一样的标准化
    tensor = torch.from_numpy((canvas - 0.1307) / 0.3081).unsqueeze(0).unsqueeze(0)
    return tensor, canvas
