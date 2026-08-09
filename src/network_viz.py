"""神经网络实时计算可视化：内联 SVG + CSS 动画展示 784 -> 128 -> 10 的计算过程。

生成的 HTML 是自包含的（含 <style>），可以直接嵌入 Gradio 页面或独立打开。
动画循环：输入像素点亮 -> 输入→隐藏层连线脉冲 -> 隐藏层神经元点亮
-> 隐藏→输出层连线脉冲 -> 输出层 0~9 共 10 个节点给出概率。
"""

import numpy as np
import torch


def build_network_html(mlp, tensor):
    """根据 MLP 的一次真实前向传播，生成神经网络计算动画的 HTML。"""
    with torch.no_grad():
        x = tensor.view(1, -1)  # 784 个输入像素
        h = torch.relu(mlp.fc1(x))  # 隐藏层 128 个神经元
        probs = torch.softmax(mlp.fc2(h), dim=1)[0].numpy()  # 输出层 10 个概率

    pixels = (tensor[0, 0].numpy() * 0.3081 + 0.1307).clip(0, 1)  # 还原 0~1 灰度
    hidden = h[0].numpy()
    h_max = max(1e-6, float(hidden.max()))

    # ---------- 布局参数 ----------
    cell = 6  # 输入层每个像素格子的边长
    inp_x, inp_y = 60, 130  # 输入层 28x28 网格左上角
    hid_x, hid_y = 470, 80  # 隐藏层节点区域左上角
    out_x, out_y = 930, 100  # 输出层第一个节点的圆心
    out_step = 36  # 输出节点间距

    # ---------- 输入层：784 个像素格子 ----------
    cells = []
    for r in range(28):
        for c in range(28):
            v = float(pixels[r, c])
            alpha = 0.12 + 0.88 * v  # 笔画越深越不透明
            delay = -round(0.55 * v, 3)  # 越亮的像素越早点亮
            cells.append(
                f'<rect x="{inp_x + c * cell}" y="{inp_y + r * cell}" '
                f'width="{cell}" height="{cell}" '
                f'fill="rgba(51,65,85,{alpha:.3f})" class="in-cell" '
                f'style="animation-delay:{delay}s"/>'
            )

    # ---------- 隐藏层：128 个神经元（8 行 x 16 列） ----------
    hid_pos = []
    hid_nodes = []
    for i in range(128):
        cx = hid_x + (i % 16) * 15
        cy = hid_y + (i // 16) * 15
        hid_pos.append((cx, cy))
        nv = float(hidden[i]) / h_max  # 归一化激活强度
        alpha = 0.12 + 0.88 * nv
        delay = -round(1.05 + 0.45 * nv, 3)  # 激活越强越早点亮
        hid_nodes.append(
            f'<circle cx="{cx}" cy="{cy}" r="6" '
            f'fill="rgba(239,68,68,{alpha:.3f})" class="hid-node" '
            f'style="animation-delay:{delay}s"/>'
        )

    # ---------- 连线：输入层 -> 隐藏层（128 条细线组成的连接束） ----------
    ih_lines = []
    x0 = inp_x + 28 * cell + 12
    for i, (cx, cy) in enumerate(hid_pos):
        y0 = inp_y + (i / 127) * 28 * cell
        ih_lines.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{cx}" y2="{cy}" '
            f'stroke="#60a5fa" stroke-width="1" class="line-ih"/>'
        )

    # ---------- 连线：隐藏层 -> 输出层（每个隐藏节点连到最近输出节点） ----------
    ho_lines = []
    for i, (cx, cy) in enumerate(hid_pos):
        oi = (i * 10) // 128
        ho_lines.append(
            f'<line x1="{cx}" y1="{cy}" x2="{out_x + 28}" y2="{out_y + oi * out_step}" '
            f'stroke="#818cf8" stroke-width="1" class="line-ho"/>'
        )

    # ---------- 输出层：0~9 共 10 个节点 ----------
    top1 = int(probs.argmax())
    out_nodes = []
    for i in range(10):
        cx = out_x
        cy = out_y + i * out_step
        p = float(probs[i])
        alpha = 0.22 + 0.78 * p  # 概率越高越亮
        delay = -round(1.75 + 0.6 * p, 3)
        ring = (
            f'<circle cx="{cx}" cy="{cy}" r="24" fill="none" stroke="#1d4ed8" '
            f'stroke-width="2" class="out-node" style="animation-delay:{delay}s"/>'
            if i == top1
            else ""
        )
        out_nodes.append(
            ring
            + f'<circle cx="{cx}" cy="{cy}" r="20" fill="rgba(37,99,235,{alpha:.3f})" '
            f'class="out-node" style="animation-delay:{delay}s"/>'
            + f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" font-size="16" '
            f'font-weight="bold" fill="#ffffff">{i}</text>'
            + f'<text x="{cx + 32}" y="{cy + 5}" font-size="13" '
            f'font-family="Consolas,monospace" '
            f'fill="{("#1d4ed8" if i == top1 else "#64748b")}" class="out-label">{p:.1%}</text>'
        )

    # ---------- 三个图层面板与标题 ----------
    panels = [
        f'<rect x="{inp_x - 20}" y="{inp_y - 48}" width="{28 * cell + 40}" '
        f'height="{28 * cell + 76}" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<rect x="{hid_x - 28}" y="{hid_y - 48}" width="{16 * 15 - 15 + 56}" '
        f'height="{8 * 15 - 15 + 86}" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>',
        f'<rect x="{out_x - 34}" y="{out_y - 62}" width="260" '
        f'height="{9 * out_step + 112}" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>',
    ]
    titles = [
        f'<text x="{inp_x + 14 * cell}" y="{inp_y - 58}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">输入层 · 28×28 = 784 个像素</text>',
        f'<text x="{hid_x + 120}" y="{hid_y - 58}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">隐藏层 · 128 个神经元</text>',
        f'<text x="{out_x + 96}" y="{out_y - 72}" text-anchor="middle" '
        f'font-size="15" font-weight="700" fill="#334155">输出层 · 0~9 十个节点</text>',
    ]

    # ---------- 底部流动指示 ----------
    flow = (
        '<text x="500" y="548" font-size="13" fill="#94a3b8">动画循环：</text>'
        '<circle cx="580" cy="543" r="8" fill="#334155" class="flow1"/>'
        '<text x="580" y="532" font-size="11" fill="#475569">输入</text>'
        '<circle cx="640" cy="543" r="8" fill="#ef4444" class="flow2"/>'
        '<text x="640" y="532" font-size="11" fill="#475569">隐藏层</text>'
        '<circle cx="700" cy="543" r="8" fill="#2563eb" class="flow3"/>'
        '<text x="700" y="532" font-size="11" fill="#475569">输出</text>'
    )

    css = """
    <style>
      .netviz-svg { width: 100%; height: auto; display: block; }
      @keyframes inK { 0%,100% { opacity:.45 } 7% { opacity:1 } 22% { opacity:.6 } }
      @keyframes ihK { 0%,100% { opacity:.05 } 28% { opacity:.55 } 46% { opacity:.08 } }
      @keyframes hidK { 0%,100% { opacity:.35 } 45% { opacity:1 } 62% { opacity:.55 } }
      @keyframes hoK { 0%,100% { opacity:.04 } 60% { opacity:.5 } 78% { opacity:.07 } }
      @keyframes outK { 0%,100% { opacity:.45 } 76% { opacity:1 } 92% { opacity:.6 } }
      @keyframes outL { 0%,68% { opacity:0 } 78% { opacity:1 } 100% { opacity:1 } }
      @keyframes flowK { 0%,100% { opacity:.2 } 35% { opacity:1 } }
      .in-cell  { animation: inK  3.2s infinite; }
      .line-ih  { animation: ihK  3.2s infinite; animation-delay:-.55s; }
      .hid-node { animation: hidK 3.2s infinite; }
      .line-ho  { animation: hoK  3.2s infinite; animation-delay:-1.35s; }
      .out-node { animation: outK 3.2s infinite; }
      .out-label{ animation: outL 3.2s infinite; animation-delay:-1.9s; }
      .flow1    { animation: flowK 3.2s infinite; animation-delay:-.2s; }
      .flow2    { animation: flowK 3.2s infinite; animation-delay:-1.3s; }
      .flow3    { animation: flowK 3.2s infinite; animation-delay:-2.4s; }
    </style>
    """

    svg = (
        f'<svg class="netviz-svg" viewBox="0 0 1150 560" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(panels)
        + "".join(ih_lines)
        + "".join(ho_lines)
        + "".join(cells)
        + "".join(hid_nodes)
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
        "✏️ 写一个数字后，神经网络会在这里实时计算："
        "输入层 → 隐藏层 → 输出层（0~9 十个节点给出概率）"
        "</div>"
    )


def save_preview(mlp, tensor, path):
    """把一次计算的动画存成独立 HTML，方便直接在浏览器里预览。"""
    from pathlib import Path

    html = build_network_html(mlp, tensor)
    Path(path).write_text(html, encoding="utf-8")
    return path
