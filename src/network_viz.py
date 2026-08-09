"""CNN 实时计算可视化（小白版）：
特征图动画 + 全连接层打分条 + Grad-CAM“为什么”证据图。

动画流程（循环约 4.2 秒）：
你写的数字 -> 卷积层1（32 张特征图）-> 卷积层2（64 张）-> 池化（64 张 7×7）
-> 全连接层投票打分（10 个数字各得一分）-> 输出层给出概率
-> 最下面的“为什么”高亮图说明模型重点看了哪里。
所有数据都是模型真实的中间计算结果，不是示意图。
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


def _gradcam_overlay(canvas, cam_14):
    """把 Grad-CAM 热力图叠加到输入图上，返回 28x28 RGB。"""
    cam = cam_14.astype(np.float32)
    cam = np.maximum(cam, 0)
    cam = cam / max(float(cam.max()), 1e-6)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L").resize(
        (28, 28), Image.LANCZOS
    )
    cam_28 = np.asarray(cam_img, dtype=np.float32) / 255.0

    base = np.stack([canvas] * 3, axis=-1)  # (28,28,3)
    heat = _hot_rgb(cam_28) / 255.0
    blend = np.clip(0.45 * base + 0.55 * heat, 0, 1)
    return (blend * 255).astype(np.uint8)


# ---------- 主入口 ----------


def build_cnn_html(cnn, tensor):
    """根据 CNN 的一次真实前向传播 + 反向传播，生成完整教学动画 HTML。"""
    # 一次前向（带梯度，用于 Grad-CAM），同时用钩子取卷积层输出
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            out.retain_grad()  # 中间层输出默认不留梯度，需要显式保留
            activations[name] = out

        return hook

    h1 = cnn.conv1.register_forward_hook(make_hook("conv1"))
    h2 = cnn.conv2.register_forward_hook(make_hook("conv2"))
    cnn.zero_grad(set_to_none=True)
    out = cnn(tensor)
    logits = out[0].detach().numpy()
    probs = torch.softmax(out[0], dim=0).detach().numpy()
    top1 = int(logits.argmax())

    # Grad-CAM：求“最可能的数字”对卷积层2输出的梯度，得到每个特征图的重要性
    out[0, top1].backward()
    grads = activations["conv2"].grad[0].detach()  # (64, 14, 14)
    h1.remove()
    h2.remove()

    a1 = torch.relu(activations["conv1"][0]).detach().numpy()  # (32, 28, 28)
    a2 = torch.relu(activations["conv2"][0]).detach().numpy()  # (64, 14, 14)
    p2 = F.max_pool2d(torch.relu(activations["conv2"][0]), 2).detach().numpy()  # (64,7,7)

    w_cam = grads.mean(dim=(1, 2))  # (64,) 每张特征图的重要性
    cam_14 = np.maximum((w_cam.numpy()[:, None, None] * a2).sum(axis=0), 0)  # (14,14)

    canvas = (tensor[0, 0].numpy() * 0.3081 + 0.1307).clip(0, 1)
    uri_in = _grayscale_uri(canvas, 120)
    uri_c1 = _to_data_uri(_feature_montage(a1, cols=8, cell=22))
    uri_c2 = _to_data_uri(_feature_montage(a2, cols=8, cell=15))
    uri_p2 = _to_data_uri(_feature_montage(p2, cols=8, cell=15))
    uri_cam = _to_data_uri(
        Image.fromarray(_gradcam_overlay(canvas, cam_14)).resize((200, 200), Image.LANCZOS)
    )

    # ---------- 第一行：流程面板 ----------
    inp_x, inp_y, inp_w, inp_h = 30, 60, 170, 300
    c1_x, c1_y, c1_w, c1_h = 240, 60, 250, 300
    c2_x, c2_y, c2_w, c2_h = 530, 60, 250, 300
    p2_x, p2_y, p2_w, p2_h = 820, 60, 250, 300
    out_x, out_y, out_w, out_h = 1110, 60, 360, 300

    panels = []
    titles = []
    for x, y, w, h in [
        (inp_x, inp_y, inp_w, inp_h),
        (c1_x, c1_y, c1_w, c1_h),
        (c2_x, c2_y, c2_w, c2_h),
        (p2_x, p2_y, p2_w, p2_h),
        (out_x, out_y, out_w, out_h),
    ]:
        panels.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="#f8fafc" stroke="#e2e8f0"/>'
        )
    titles = [
        f'<text x="{inp_x + inp_w // 2}" y="{inp_y + 26}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="#334155">① 你写的数字 · 28×28</text>',
        f'<text x="{c1_x + c1_w // 2}" y="{c1_y + 26}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="#334155">② 找小模板 · 32 张特征图</text>',
        f'<text x="{c2_x + c2_w // 2}" y="{c2_y + 26}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="#334155">③ 组合成笔画 · 64 张</text>',
        f'<text x="{p2_x + p2_w // 2}" y="{p2_y + 26}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="#334155">④ 压缩 · 64 张 7×7</text>',
        f'<text x="{out_x + out_w // 2}" y="{out_y + 26}" text-anchor="middle" '
        f'font-size="14" font-weight="700" fill="#334155">⑥ 结果：谁分高谁是答案</text>',
    ]

    input_img = (
        f'<image href="{uri_in}" x="{inp_x + (inp_w - 120) // 2}" y="{inp_y + 55}" '
        f'width="120" height="120" class="in-img"/>'
        f'<text x="{inp_x + inp_w // 2}" y="{inp_y + 210}" text-anchor="middle" '
        f'font-size="12" fill="#64748b">每个格子 = 一个像素的明暗</text>'
    )

    def _stage(uri, panel_x, panel_w, img_w, img_h, cls, caption, cy):
        x = panel_x + (panel_w - img_w) // 2
        return (
            f'<image href="{uri}" x="{x}" y="{cy}" width="{img_w}" '
            f'height="{img_h}" class="{cls}"/>'
            f'<text x="{panel_x + panel_w // 2}" y="{cy + img_h + 20}" '
            f'text-anchor="middle" font-size="12" fill="#64748b">{caption}</text>'
        )

    stages = (
        _stage(uri_c1, c1_x, c1_w, 194, 98, "fm1", "越亮 = 越像某个小模板", 130)
        + _stage(uri_c2, c2_x, c2_w, 138, 138, "fm2", "组合成横、竖、圈等", 130)
        + _stage(uri_p2, p2_x, p2_w, 138, 138, "fm3", "只留最明显的特征", 130)
    )

    # 第一行：输入 -> 卷积1 -> 卷积2 -> 池化 -> 输出 的箭头
    arrows = []
    for ax, cls in [(220, "arrow1"), (500, "arrow2"), (790, "arrow3"), (1080, "arrow4")]:
        arrows.append(
            f'<g class="{cls}">'
            f'<line x1="{ax - 12}" y1="210" x2="{ax + 10}" y2="210" '
            f'stroke="#94a3b8" stroke-width="3" stroke-linecap="round"/>'
            f'<polygon points="{ax + 14},210 {ax + 4},204 {ax + 4},216" fill="#94a3b8"/>'
            "</g>"
        )

    # 输出层：10 个节点 + 概率
    out_nodes = []
    for i in range(10):
        cx = out_x + 50
        cy = out_y + 45 + i * 24
        p = float(probs[i])
        alpha = 0.22 + 0.78 * p
        delay = -round(3.0 + 0.3 * p, 3)
        ring = (
            f'<circle cx="{cx}" cy="{cy}" r="17" fill="none" stroke="#1d4ed8" '
            f'stroke-width="2" class="out-node" style="animation-delay:{delay}s"/>'
            if i == top1
            else ""
        )
        out_nodes.append(
            ring
            + f'<circle cx="{cx}" cy="{cy}" r="14" fill="rgba(37,99,235,{alpha:.3f})" '
            f'class="out-node" style="animation-delay:{delay}s"/>'
            + f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="13" '
            f'font-weight="bold" fill="#ffffff">{i}</text>'
            + f'<text x="{cx + 24}" y="{cy + 5}" font-size="12" '
            f'font-family="Consolas,monospace" '
            f'fill="{("#1d4ed8" if i == top1 else "#64748b")}" class="out-label">{p:.1%}</text>'
        )
    out_nodes.append(
        f'<text x="{out_x + out_w // 2}" y="{out_y + 278}" text-anchor="middle" '
        f'font-size="12" fill="#64748b">软归一化：分数 → 概率</text>'
    )

    # ---------- 第二行：投票打分面板 ----------
    vote_x, vote_y, vote_w, vote_h = 30, 400, 760, 270
    max_abs = max(float(np.abs(logits).max()), 1e-6)
    bar_max_w = 300
    zero_x = vote_x + 300
    bars = []
    for i in range(10):
        s = float(logits[i])
        w = abs(s) / max_abs * bar_max_w
        cy = vote_y + 52 + i * 21
        is_top = i == top1
        if s >= 0:
            x0, origin = zero_x, "0% 50%"
            color = "#2563eb" if is_top else "#93c5fd"
        else:
            x0, origin = zero_x - w, "100% 50%"
            color = "#94a3b8"
        bars.append(
            f'<text x="{vote_x + 30}" y="{cy + 4}" text-anchor="middle" font-size="12" '
            f'font-weight="bold" fill="{("#1d4ed8" if is_top else "#475569")}">{i}</text>'
            f'<rect x="{x0:.1f}" y="{cy - 8}" width="{w:.1f}" height="16" rx="4" '
            f'fill="{color}" class="score-bar" '
            f'style="animation-delay:-2.35s;transform-box:fill-box;transform-origin:{origin}"/>'
            f'<text x="{x0 + (8 if s >= 0 else -8)}" y="{cy + 4}" font-size="11" '
            f'font-family="Consolas,monospace" fill="#475569" text-anchor="'
            f'{("start" if s >= 0 else "end")}" class="score-label">{s:+.2f}</text>'
        )
    bars.append(
        f'<line x1="{zero_x}" y1="{vote_y + 40}" x2="{zero_x}" y2="{vote_y + 245}" '
        f'stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<text x="{zero_x}" y="{vote_y + 262}" text-anchor="middle" font-size="12" '
        f'fill="#64748b">0 分线</text>'
    )
    vote_title = (
        f'<text x="{vote_x + vote_w // 2}" y="{vote_y + 30}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">⑤ 全连接层投票打分</text>'
        f'<text x="{vote_x + vote_w // 2}" y="{vote_y + 48}" text-anchor="middle" '
        f'font-size="12" fill="#64748b">得分 = 每个特征 × 权重 加总（正 = 支持，负 = 反对）</text>'
    )

    # ---------- 第二行：Grad-CAM 为什么面板 ----------
    cam_x, cam_y, cam_w, cam_h = 820, 400, 650, 270
    cam_img = (
        f'<image href="{uri_cam}" x="{cam_x + 30}" y="{cam_y + 55}" '
        f'width="200" height="200" class="grad-img"/>'
        f'<text x="{cam_x + 130}" y="{cam_y + 275}" text-anchor="middle" '
        f'font-size="12" fill="#64748b">红/黄越亮 = 模型越看重这块</text>'
    )
    cam_text = (
        f'<text x="{cam_x + 260}" y="{cam_y + 80}" font-size="14" fill="#334155">'
        f'🎯 模型认为这是 {top1}，因为：</text>'
        f'<text x="{cam_x + 260}" y="{cam_y + 108}" font-size="13" fill="#475569">'
        f'它把判断依据“映射”回原图，</text>'
        f'<text x="{cam_x + 260}" y="{cam_y + 130}" font-size="13" fill="#475569">'
        f'亮的地方就是它重点看的区域。</text>'
        f'<text x="{cam_x + 260}" y="{cam_y + 162}" font-size="12" fill="#64748b">'
        f'（专业叫法：Grad-CAM 热力图）</text>'
    )
    cam_title = (
        f'<text x="{cam_x + cam_w // 2}" y="{cam_y + 30}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#b45309">为什么判断是 {top1}（证据图）</text>'
    )

    # ---------- 底部流动指示 ----------
    flow = (
        '<text x="560" y="688" font-size="13" fill="#94a3b8">流程：</text>'
        '<circle cx="620" cy="683" r="7" fill="#334155" class="flow1"/>'
        '<text x="620" y="671" font-size="11" fill="#475569">看</text>'
        '<circle cx="675" cy="683" r="7" fill="#ef4444" class="flow2"/>'
        '<text x="675" y="671" font-size="11" fill="#475569">找模板</text>'
        '<circle cx="730" cy="683" r="7" fill="#ef4444" class="flow3"/>'
        '<text x="730" y="671" font-size="11" fill="#475569">组合</text>'
        '<circle cx="785" cy="683" r="7" fill="#ef4444" class="flow4"/>'
        '<text x="785" y="671" font-size="11" fill="#475569">压缩</text>'
        '<circle cx="840" cy="683" r="7" fill="#f59e0b" class="flow5"/>'
        '<text x="840" y="671" font-size="11" fill="#475569">投票</text>'
        '<circle cx="895" cy="683" r="7" fill="#2563eb" class="flow6"/>'
        '<text x="895" y="671" font-size="11" fill="#475569">出结果</text>'
        '<text x="1010" y="688" font-size="12" fill="#94a3b8">看 → 找模板 → 组合 → 压缩 → 投票 → 出结果</text>'
    )

    css = """
    <style>
      .netviz-svg { width: 100%; height: auto; display: block; }
      @keyframes inK   { 0%,100% { opacity:.5 } 5%  { opacity:1 } 18% { opacity:.65 } }
      @keyframes arK   { 0%,100% { opacity:.12 } 14% { opacity:.9 } 30% { opacity:.2 } }
      @keyframes fmK   { 0%,100% { opacity:.15 } 26% { opacity:1 } 45% { opacity:.5 } }
      @keyframes barK  { 0%,55% { transform:scaleX(0); opacity:.2 } 62% { transform:scaleX(1); opacity:1 }
                         85% { transform:scaleX(1); opacity:1 } 100% { transform:scaleX(0); opacity:.2 } }
      @keyframes outK  { 0%,100% { opacity:.45 } 72% { opacity:1 } 90% { opacity:.6 } }
      @keyframes outL  { 0%,68% { opacity:0 } 76% { opacity:1 } 100% { opacity:1 } }
      @keyframes gradK { 0%,60% { opacity:0 } 70% { opacity:1 } 100% { opacity:1 } }
      @keyframes flowK { 0%,100% { opacity:.2 } 35% { opacity:1 } }
      .in-img   { animation: inK   4.2s infinite; animation-delay:-.15s; }
      .arrow1   { animation: arK   4.2s infinite; animation-delay:-.5s; }
      .fm1      { animation: fmK   4.2s infinite; animation-delay:-.7s; }
      .arrow2   { animation: arK   4.2s infinite; animation-delay:-1.05s; }
      .fm2      { animation: fmK   4.2s infinite; animation-delay:-1.25s; }
      .arrow3   { animation: arK   4.2s infinite; animation-delay:-1.6s; }
      .fm3      { animation: fmK   4.2s infinite; animation-delay:-1.8s; }
      .arrow4   { animation: arK   4.2s infinite; animation-delay:-2.15s; }
      .score-bar{ animation: barK  4.2s infinite; }
      .score-label{ animation: outL 4.2s infinite; animation-delay:-2.55s; }
      .out-node { animation: outK  4.2s infinite; }
      .out-label{ animation: outL  4.2s infinite; animation-delay:-3.2s; }
      .grad-img { animation: gradK 4.2s infinite; animation-delay:-2.75s; }
      .flow1    { animation: flowK 4.2s infinite; animation-delay:-.2s; }
      .flow2    { animation: flowK 4.2s infinite; animation-delay:-.9s; }
      .flow3    { animation: flowK 4.2s infinite; animation-delay:-1.6s; }
      .flow4    { animation: flowK 4.2s infinite; animation-delay:-2.3s; }
      .flow5    { animation: flowK 4.2s infinite; animation-delay:-2.95s; }
      .flow6    { animation: flowK 4.2s infinite; animation-delay:-3.5s; }
    </style>
    """

    svg = (
        f'<svg class="netviz-svg" viewBox="0 0 1500 700" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(panels)
        + "".join(arrows)
        + input_img
        + stages
        + "".join(out_nodes)
        + vote_title
        + "".join(bars)
        + cam_title
        + cam_img
        + cam_text
        + "".join(titles)
        + flow
        + "</svg>"
    )
    return f"<div style='width:100%;'>{css}{svg}</div>"


def build_empty_html():
    """没有输入时的占位可视化。"""
    return (
        "<div style='width:100%;display:flex;align-items:center;justify-content:center;"
        "height:380px;background:#f8fafc;border-radius:14px;color:#94a3b8;"
        "font-size:16px;'>"
        "✏️ 写一个数字后，这里会完整演示："
        "看 → 找模板 → 压缩 → 投票打分 → 出结果，最后告诉你模型为什么这么判断"
        "</div>"
    )


def save_preview(cnn, tensor, path):
    """把一次计算的动画存成独立 HTML，方便直接在浏览器里预览。"""
    from pathlib import Path

    Path(path).write_text(build_cnn_html(cnn, tensor), encoding="utf-8")
    return path
