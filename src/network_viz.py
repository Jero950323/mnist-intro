"""CNN 实时计算可视化：特征图动画 + 输出层概率（内联 SVG + base64 特征图 + CSS 动画）。

动画流程（循环约 3.6 秒）：
输入 28×28 -> 卷积层 1（32 张特征图点亮）-> 卷积层 2（64 张点亮）
-> 池化后（64 张 7×7）-> 全连接 -> 输出层 0~9 共 10 个节点给出概率。
特征图是模型真实的中间计算结果，不是示意图。
"""

import base64
import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------- 图像工具 ----------


def _hot_rgb(norm):
    """把 0~1 强度映射成 hot 风格彩色（黑 -> 红 -> 黄 -> 白）。"""
    r = np.clip(norm * 3.0, 0, 1)
    g = np.clip(norm * 3.0 - 1.0, 0, 1)
    b = np.clip(norm * 3.0 - 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _feature_montage(maps, cols, cell):
    """把 N 张特征图拼成一张缩略图墙。maps: (N, H, W) numpy（已过 ReLU）。"""
    n = maps.shape[0]
    rows = math.ceil(n / cols)
    gap = 2
    canvas = Image.new(
        "RGB",
        (cols * (cell + gap) + gap, rows * (cell + gap) + gap),
        (10, 14, 26),
    )
    for i in range(n):
        m = maps[i]
        vmax = max(1e-6, float(m.max()))
        rgb = _hot_rgb(m / vmax)
        img = Image.fromarray(rgb).resize((cell, cell), Image.LANCZOS)
        r_, c_ = divmod(i, cols)
        canvas.paste(img, (gap + c_ * (cell + gap), gap + r_ * (cell + gap)))
    return canvas


def _to_data_uri(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _grayscale_uri(canvas, size):
    """28x28 灰度 -> 放大后的 PNG data URI。"""
    img = Image.fromarray((canvas * 255).astype(np.uint8), mode="L").resize(
        (size, size), Image.LANCZOS
    )
    return _to_data_uri(img)


# ---------- 主入口 ----------


def build_cnn_html(cnn, tensor):
    """根据 CNN 的一次真实前向传播，生成特征图动画 + 输出概率的 HTML。"""
    with torch.no_grad():
        probs = torch.softmax(cnn(tensor), dim=1)[0].numpy()

    # 用钩子取卷积层输出（真实中间结果）
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()

        return hook

    h1 = cnn.conv1.register_forward_hook(make_hook("conv1"))
    h2 = cnn.conv2.register_forward_hook(make_hook("conv2"))
    with torch.no_grad():
        cnn(tensor)
    h1.remove()
    h2.remove()

    a1 = torch.relu(activations["conv1"][0]).numpy()  # (32, 28, 28)
    a2 = torch.relu(activations["conv2"][0]).numpy()  # (64, 14, 14)
    p2 = torch.relu(
        F.max_pool2d(torch.from_numpy(a2).unsqueeze(0), 2)[0]
    ).numpy()  # (64, 7, 7)

    canvas = (tensor[0, 0].numpy() * 0.3081 + 0.1307).clip(0, 1)

    uri_in = _grayscale_uri(canvas, 130)
    uri_c1 = _to_data_uri(_feature_montage(a1, cols=8, cell=22))
    uri_c2 = _to_data_uri(_feature_montage(a2, cols=8, cell=15))
    uri_p2 = _to_data_uri(_feature_montage(p2, cols=8, cell=15))

    # ---------- 布局参数 ----------
    inp_x, inp_y, inp_w, inp_h = 30, 70, 190, 300
    c1_x, c1_y, c1_w, c1_h = 280, 70, 250, 300
    c2_x, c2_y, c2_w, c2_h = 560, 70, 250, 300
    p2_x, p2_y, p2_w, p2_h = 840, 70, 250, 300
    out_x, out_y, out_w, out_h = 1110, 70, 170, 330
    out_step = 30

    # ---------- 面板 ----------
    panels = [
        f'<rect x="{inp_x}" y="{inp_y}" width="{inp_w}" height="{inp_h}" rx="12" '
        f'fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<rect x="{c1_x}" y="{c1_y}" width="{c1_w}" height="{c1_h}" rx="12" '
        f'fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<rect x="{c2_x}" y="{c2_y}" width="{c2_w}" height="{c2_h}" rx="12" '
        f'fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="12" '
        f'fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<rect x="{out_x}" y="{out_y}" width="{out_w}" height="{out_h}" rx="12" '
        f'fill="#f8fafc" stroke="#e2e8f0"/>',
    ]

    # ---------- 标题 ----------
    titles = [
        f'<text x="{inp_x + inp_w // 2}" y="{inp_y + 28}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">输入层 · 28×28</text>',
        f'<text x="{c1_x + c1_w // 2}" y="{c1_y + 28}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">卷积层 1 · 32 个特征图</text>',
        f'<text x="{c2_x + c2_w // 2}" y="{c2_y + 28}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">卷积层 2 · 64 个特征图</text>',
        f'<text x="{p2_x + p2_w // 2}" y="{p2_y + 28}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">池化后 · 64 张 7×7</text>',
        f'<text x="{out_x + out_w // 2}" y="{out_y + 28}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">输出层 · 0~9</text>',
    ]

    # ---------- 内容 ----------
    input_img = (
        f'<image href="{uri_in}" x="{inp_x + (inp_w - 130) // 2}" y="{inp_y + 55}" '
        f'width="130" height="130" class="in-img"/>'
    )

    def _stage(uri, panel_x, panel_w, img_w, img_h, cls, caption, cy):
        x = panel_x + (panel_w - img_w) // 2
        return (
            f'<image href="{uri}" x="{x}" y="{cy}" width="{img_w}" '
            f'height="{img_h}" class="{cls}"/>'
            f'<text x="{panel_x + panel_w // 2}" y="{cy + img_h + 22}" '
            f'text-anchor="middle" font-size="12" fill="#64748b">{caption}</text>'
        )

    stages = (
        _stage(uri_c1, c1_x, c1_w, 194, 98, "fm1", "每张 28×28", 140)
        + _stage(uri_c2, c2_x, c2_w, 138, 138, "fm2", "每张 14×14", 140)
        + _stage(uri_p2, p2_x, p2_w, 138, 138, "fm3", "每张 7×7", 140)
    )

    # ---------- 箭头 ----------
    arrows = []
    for ax, cls in [(250, "arrow1"), (545, "arrow2"), (825, "arrow3"), (1100, "arrow4")]:
        arrows.append(
            f'<g class="{cls}">'
            f'<line x1="{ax - 12}" y1="220" x2="{ax + 10}" y2="220" '
            f'stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>'
            f'<polygon points="{ax + 14},220 {ax + 4},214 {ax + 4},226" fill="#94a3b8"/>'
            "</g>"
        )

    # ---------- 输出层节点 ----------
    top1 = int(probs.argmax())
    out_nodes = []
    for i in range(10):
        cx = out_x + out_w // 2
        cy = out_y + 55 + i * out_step
        p = float(probs[i])
        alpha = 0.22 + 0.78 * p
        delay = -round(2.55 + 0.45 * p, 3)
        ring = (
            f'<circle cx="{cx}" cy="{cy}" r="22" fill="none" stroke="#1d4ed8" '
            f'stroke-width="2" class="out-node" style="animation-delay:{delay}s"/>'
            if i == top1
            else ""
        )
        out_nodes.append(
            ring
            + f'<circle cx="{cx}" cy="{cy}" r="18" fill="rgba(37,99,235,{alpha:.3f})" '
            f'class="out-node" style="animation-delay:{delay}s"/>'
            + f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="15" '
            f'font-weight="bold" fill="#ffffff">{i}</text>'
            + f'<text x="{cx + 26}" y="{cy + 5}" font-size="12" '
            f'font-family="Consolas,monospace" '
            f'fill="{("#1d4ed8" if i == top1 else "#64748b")}" class="out-label">{p:.1%}</text>'
        )

    # ---------- 底部流动指示 ----------
    flow = (
        '<text x="470" y="548" font-size="13" fill="#94a3b8">动画循环：</text>'
        '<circle cx="545" cy="543" r="7" fill="#334155" class="flow1"/>'
        '<text x="545" y="531" font-size="11" fill="#475569">输入</text>'
        '<circle cx="600" cy="543" r="7" fill="#ef4444" class="flow2"/>'
        '<text x="600" y="531" font-size="11" fill="#475569">卷积1</text>'
        '<circle cx="655" cy="543" r="7" fill="#ef4444" class="flow3"/>'
        '<text x="655" y="531" font-size="11" fill="#475569">卷积2</text>'
        '<circle cx="710" cy="543" r="7" fill="#ef4444" class="flow4"/>'
        '<text x="710" y="531" font-size="11" fill="#475569">池化</text>'
        '<circle cx="765" cy="543" r="7" fill="#2563eb" class="flow5"/>'
        '<text x="765" y="531" font-size="11" fill="#475569">输出</text>'
    )

    css = """
    <style>
      .netviz-svg { width: 100%; height: auto; display: block; }
      @keyframes inK  { 0%,100% { opacity:.45 } 6%  { opacity:1 } 22% { opacity:.6 } }
      @keyframes arK  { 0%,100% { opacity:.12 } 16% { opacity:.9 } 34% { opacity:.2 } }
      @keyframes fmK  { 0%,100% { opacity:.15 } 28% { opacity:1 } 48% { opacity:.5 } }
      @keyframes outK { 0%,100% { opacity:.45 } 72% { opacity:1 } 90% { opacity:.6 } }
      @keyframes outL { 0%,66% { opacity:0 } 74% { opacity:1 } 100% { opacity:1 } }
      @keyframes flowK{ 0%,100% { opacity:.2 } 35% { opacity:1 } }
      .in-img  { animation: inK   3.6s infinite; animation-delay:-.1s; }
      .arrow1  { animation: arK   3.6s infinite; animation-delay:-.45s; }
      .fm1     { animation: fmK   3.6s infinite; animation-delay:-.65s; }
      .arrow2  { animation: arK   3.6s infinite; animation-delay:-1.0s; }
      .fm2     { animation: fmK   3.6s infinite; animation-delay:-1.2s; }
      .arrow3  { animation: arK   3.6s infinite; animation-delay:-1.55s; }
      .fm3     { animation: fmK   3.6s infinite; animation-delay:-1.75s; }
      .arrow4  { animation: arK   3.6s infinite; animation-delay:-2.15s; }
      .out-node{ animation: outK  3.6s infinite; }
      .out-label{animation: outL  3.6s infinite; animation-delay:-2.75s; }
      .flow1   { animation: flowK 3.6s infinite; animation-delay:-.2s; }
      .flow2   { animation: flowK 3.6s infinite; animation-delay:-.95s; }
      .flow3   { animation: flowK 3.6s infinite; animation-delay:-1.7s; }
      .flow4   { animation: flowK 3.6s infinite; animation-delay:-2.45s; }
      .flow5   { animation: flowK 3.6s infinite; animation-delay:-3.0s; }
    </style>
    """

    svg = (
        f'<svg class="netviz-svg" viewBox="0 0 1280 580" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(panels)
        + "".join(arrows)
        + input_img
        + stages
        + "".join(out_nodes)
        + "".join(titles)
        + flow
        + "</svg>"
    )
    return f"<div style='width:100%;'>{css}{svg}</div>"


def build_empty_html():
    """没有输入时的占位可视化。"""
    return (
        "<div style='width:100%;display:flex;align-items:center;justify-content:center;"
        "height:340px;background:#f8fafc;border-radius:14px;color:#94a3b8;"
        "font-size:16px;'>"
        "✏️ 写一个数字后，CNN 会在这里实时计算："
        "输入图片 → 卷积特征图 → 输出层 0~9 十个节点给出概率"
        "</div>"
    )


def save_preview(cnn, tensor, path):
    """把一次计算的动画存成独立 HTML，方便直接在浏览器里预览。"""
    from pathlib import Path

    Path(path).write_text(build_cnn_html(cnn, tensor), encoding="utf-8")
    return path
