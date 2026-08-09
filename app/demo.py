"""Gradio 交互演示：用鼠标手写数字，模型实时识别。

用法（在项目根目录）:
    python app/demo.py
    或
    python app/demo.py --share    # 生成公网链接，方便远程演示
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

# 限制数值库线程数，避免低内存机器上 OpenBLAS 分配失败
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # 服务器端绘图，不弹窗
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.model import build_model
from src.preprocess import preprocess_digit


# ---------- 模型加载 ----------


def load_model(model_name, checkpoint):
    """加载训练好的模型；如果不存在，就先用 2 个 epoch 快速训练一个。"""
    model = build_model(model_name)
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        print(f"[提示] 未找到模型 {ckpt}，先快速训练 2 个 epoch（约 1~2 分钟）……")
        subprocess.run(
            [sys.executable, "-m", "src.train", "--model", model_name, "--epochs", "2"],
            cwd=str(ROOT),
            check=True,
        )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


# ---------- 图像预处理 ----------


# ---------- 页面渲染 ----------


def render_bars(probs):
    """把 10 个数字的概率渲染成横向条形图（HTML）。"""
    order = np.argsort(probs)[::-1]
    items = []
    for rank, idx in enumerate(order):
        p = probs[idx]
        width = max(2.0, p * 100)
        bar_color = "#2563eb" if rank == 0 else "#b6ccf7"
        text_color = "#111827" if rank == 0 else "#6b7280"
        items.append(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
              <span style="width:22px;font-weight:700;color:{text_color};text-align:center;">{idx}</span>
              <div style="flex:1;background:#eef2f7;border-radius:8px;height:20px;overflow:hidden;">
                <div style="width:{width:.1f}%;height:100%;background:{bar_color};border-radius:8px;transition:width .15s;"></div>
              </div>
              <span style="width:54px;text-align:right;font-family:Consolas,monospace;color:{text_color};">{p*100:.1f}%</span>
            </div>
            """
        )
    return (
        "<div style='font-family:system-ui,-apple-system,sans-serif;'>"
        + "".join(items)
        + "</div>"
    )


def render_top(top1, conf):
    return f"""
    <div style="text-align:center;padding:6px 0;">
      <div style="font-size:16px;color:#64748b;font-weight:600;">识别结果</div>
      <div style="font-size:76px;font-weight:800;color:#1d4ed8;line-height:1.15;">{top1}</div>
      <div style="font-size:15px;color:#475569;">置信度 {conf:.1%}</div>
    </div>
    """


def classify(model, img):
    """把画板输入转成 28x28 并预测，返回 (probs, canvas, tensor)。

    输入为空时返回 (None, None, None)。probs 是 10 个数字的概率，
    canvas 是 0~1 的 28x28 灰度图（页面展示“模型看到的样子”）。
    """
    if img is None:
        return None, None, None

    # Gradio 6 的 Sketchpad 会传入 dict（background/layers/composite），
    # 旧版本则直接传 numpy 数组，这里兼容两种格式
    if isinstance(img, dict):
        composite = img.get("composite")
        img = composite if composite is not None else img.get("background")
    if img is None:
        return None, None, None

    result = preprocess_digit(img)
    if result is None:
        return None, None, None

    tensor, canvas = result
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    return probs, canvas, tensor


def predict(model, img):
    """把画板上的图像转成 28x28，交给模型预测，返回页面组件需要的内容。"""
    empty_top = (
        "<div style='text-align:center;color:#94a3b8;font-size:18px;padding:30px 0;'>"
        "请先在画板上写一个数字</div>"
    )
    empty_bars = (
        "<div style='text-align:center;color:#94a3b8;padding:12px;'>"
        "写完之后，这里会显示 0~9 每个数字的概率</div>"
    )
    probs, canvas, _ = classify(model, img)
    if probs is None:
        return empty_top, empty_bars, "", None

    top1 = int(probs.argmax())
    conf = float(probs[top1])
    msg = (
        f"识别结果：**{top1}**（置信度 {conf:.1%}）"
        "　提示：写大、写粗、写中间最准；"
        f"{'当前置信度较低，可以重写一次试试。' if conf < 0.4 else '左侧的条形图是 0~9 每个数字的概率。'}"
    )
    return (
        render_top(top1, conf),
        render_bars(probs),
        msg,
        (canvas * 255).astype(np.uint8),
    )


def render_network(mlp, tensor):
    """把 MLP 的实时计算过程画出来：输入 → 隐藏层激活 → 10 个输出概率。"""
    import matplotlib.pyplot as plt

    with torch.no_grad():
        x = tensor.view(1, -1)  # 784 个输入像素
        h = torch.relu(mlp.fc1(x))  # 隐藏层 128 个神经元
        logits = mlp.fc2(h)  # 输出层 10 个节点
        probs = torch.softmax(logits, dim=1)[0].numpy()

    canvas = tensor[0, 0].numpy() * 0.3081 + 0.1307  # 还原回 0~1 灰度
    hid = h[0].numpy()

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1, 1.4, 1.15]}
    )

    # 1) 输入层：28x28 像素
    axes[0].imshow(canvas, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("输入层\n784 个像素", fontsize=11)
    axes[0].axis("off")

    # 2) 隐藏层：128 个神经元的激活值（越亮 = 被激活得越强）
    grid = hid.reshape(8, 16)
    vmax = max(1e-6, float(hid.max()))
    im = axes[1].imshow(grid, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_title("隐藏层\n128 个神经元（实时激活值）", fontsize=11)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) 输出层：0~9 共 10 个节点，按概率排序点亮
    top1 = int(probs.argmax())
    colors = ["#2563eb" if i == top1 else "#cbd5e1" for i in range(10)]
    axes[2].barh(range(10), probs, color=colors)
    axes[2].set_yticks(range(10))
    axes[2].set_yticklabels([str(i) for i in range(10)], fontsize=12, fontweight="bold")
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_title("输出层\n0~9 共 10 个节点（概率）", fontsize=11)
    axes[2].set_xlabel("概率")
    for i, p in enumerate(probs):
        axes[2].text(p + 0.01, i, f"{p:.1%}", va="center", fontsize=9)

    fig.suptitle(
        "神经网络计算过程：手写输入 → 每一层逐步计算 → 10 个数字的概率",
        fontsize=13,
        y=1.03,
    )
    fig.tight_layout()
    return fig


def blank_network_fig():
    """没有输入时展示的占位图。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, 4.3))
    ax.text(
        0.5,
        0.5,
        "写一个数字后，这里会展示神经网络每一层的实时计算",
        ha="center",
        va="center",
        fontsize=14,
        color="#94a3b8",
    )
    ax.axis("off")
    return fig


def load_challenges(examples_dir):
    """读取挑战题（模型会认错的疑难样本），返回 [(图片路径, 正确答案), ...]。"""
    examples_dir = Path(examples_dir)
    meta_path = examples_dir / "challenges.json"
    if not meta_path.exists():
        return []
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    items = []
    for name, info in meta.items():
        p = examples_dir / name
        if p.exists():
            items.append((str(p), int(info["true"])))
    return items


# ---------- 页面组件 ----------


def build_sketchpad():
    """创建手写画板组件，兼容不同版本的 Gradio。

    Gradio 5 及之前用 shape/brush_radius；
    Gradio 6 改成了 height/width + Brush 对象。
    """
    import inspect

    params = inspect.signature(gr.Sketchpad.__init__).parameters
    if "shape" in params:
        return gr.Sketchpad(
            shape=(420, 420),
            image_mode="L",
            invert_colors=True,
            brush_radius=26,
            label="在这里用鼠标写一个 0~9 的数字",
        )
    return gr.Sketchpad(
        height=420,
        width=420,
        image_mode="L",
        canvas_size=(560, 560),
        brush=gr.Brush(default_size=26),
        label="在这里用鼠标写一个 0~9 的数字",
    )


CSS = """
.gradio-container {
  background: linear-gradient(135deg, #f6f8fc 0%, #eef2ff 100%);
  max-width: 1120px !important;
  margin: 0 auto;
}
#app-header { text-align: center; padding: 10px 0 6px; }
#app-header h1 { font-size: 28px; font-weight: 800; color: #1e293b; margin: 0; }
#app-header p { color: #64748b; margin: 6px 0 0; font-size: 14px; }
.draw-card, .result-card {
  background: #ffffff;
  border-radius: 18px;
  box-shadow: 0 6px 24px rgba(30, 64, 175, 0.08);
  padding: 18px;
}
.challenge-card {
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 18px;
  box-shadow: 0 6px 24px rgba(180, 83, 9, 0.08);
  padding: 14px 18px;
}
#tips { color: #64748b; font-size: 13px; text-align: center; margin-top: 6px; }
footer { display: none !important; }
"""


def main():
    parser = argparse.ArgumentParser(description="MNIST 手写识别交互演示")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--checkpoint", default="outputs/cnn_mnist.pth")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    model = load_model(args.model, args.checkpoint)
    mlp_model = load_model("mlp", "outputs/mlp_mnist.pth")

    example_paths = [
        str(p) for p in sorted((ROOT / "outputs" / "examples").glob("digit_*.png"))
    ]
    challenges = load_challenges(ROOT / "outputs" / "examples")

    empty_top = (
        "<div style='text-align:center;color:#94a3b8;font-size:18px;padding:30px 0;'>"
        "请先在画板上写一个数字</div>"
    )
    empty_bars = (
        "<div style='text-align:center;color:#94a3b8;padding:12px;'>"
        "写完之后，这里会显示 0~9 每个数字的概率</div>"
    )

    def handle(img):
        top, bars, msg, canvas = predict(model, img)
        _, _, tensor = classify(model, img)
        fig = render_network(mlp_model, tensor) if tensor is not None else blank_network_fig()
        return top, bars, msg, canvas, fig

    def random_example():
        return gr.update(value=random.choice(example_paths))

    def on_challenge(path, true_label):
        """挑战题：识别后揭晓答案，讲解模型的盲区。"""
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        probs, canvas, tensor = classify(model, img)
        if probs is None:
            return (
                gr.update(value=path),
                empty_top,
                empty_bars,
                "",
                None,
                blank_network_fig(),
            )
        top1 = int(probs.argmax())
        conf = float(probs[top1])
        if true_label == top1:
            reveal = f"答案就是 **{top1}**，这次模型认对了（置信度 {conf:.1%}）。"
        else:
            reveal = (
                f"答案是 **{true_label}**，模型认成了 **{top1}**（置信度 {conf:.1%}）。"
                "这就是模型的盲区：训练数据里长得像 "
                f"{top1} 的 {true_label}，模型就学会了这种对应。"
            )
        return (
            gr.update(value=path),
            render_top(top1, conf),
            render_bars(probs),
            reveal,
            (canvas * 255).astype(np.uint8),
            render_network(mlp_model, tensor),
        )

    # Gradio 6 把 css/theme 参数移到了 launch()，这里做版本兼容
    gradio_major = int(gr.__version__.split(".")[0])
    blocks_kwargs = {"title": "手写数字识别 · 深度学习原理演示"}
    launch_kwargs = {"share": args.share}
    if gradio_major >= 6:
        launch_kwargs.update(css=CSS, theme=gr.themes.Soft(primary_hue="blue"))
    else:
        blocks_kwargs.update(css=CSS, theme=gr.themes.Soft(primary_hue="blue"))

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.HTML(
            """
            <div id="app-header">
              <h1>✏️ 手写数字识别 · 深度学习原理演示</h1>
              <p>用鼠标写一个 0~9 的数字，模型实时告诉你它认为是几 —— 全程本地运行，数据不会上传</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes=["draw-card"]):
                sketch = build_sketchpad()
                with gr.Row():
                    btn_random = gr.Button("🎲 随机示例", variant="secondary")
                gr.Markdown(
                    "**提示**：写大、写粗、写中间最准；“1”直接画一根竖线。",
                    elem_id="tips",
                )

            with gr.Column(scale=4, elem_classes=["result-card"]):
                top_html = gr.HTML(
                    "<div style='text-align:center;color:#94a3b8;font-size:18px;"
                    "padding:30px 0;'>请先在画板上写一个数字</div>"
                )
                bars_html = gr.HTML(
                    "<div style='text-align:center;color:#94a3b8;padding:12px;'>"
                    "写完之后，这里会显示 0~9 每个数字的概率</div>"
                )
                with gr.Row():
                    small_img = gr.Image(
                        label="模型看到的 28×28（预处理后）",
                        height=170,
                        width=170,
                        interactive=False,
                    )
                    msg = gr.Markdown("")

        with gr.Column(elem_classes=["result-card"]):
            gr.Markdown(
                "### 🔍 神经网络内部是怎么算的\n"
                "左边是输入图片，中间是隐藏层 **128 个神经元**的实时激活值"
                "（越亮 = 被激活得越强），右边是 **0~9 共 10 个输出节点**的概率。"
                "这里用结构透明的 MLP 做演示；页面识别用的仍是更准的 CNN。"
            )
            net_fig = gr.Plot(
                label="神经网络实时计算（MLP：784 → 128 → 10）",
            )

        gr.Examples(
            examples=[[p] for p in example_paths],
            inputs=sketch,
            label="示例（点一个试试）",
        )

        if challenges:
            with gr.Column(elem_classes=["challenge-card"]):
                gr.Markdown(
                    "### 🧠 挑战题：模型的盲区\n"
                    "先猜猜这些数字是几，再点「识别」。模型答错的题，最能说明"
                    "**训练数据长什么样，模型就学什么样**。"
                )
                for i in range(0, len(challenges), 6):
                    with gr.Row():
                        for path, true_label in challenges[i : i + 6]:
                            with gr.Column(min_width=95):
                                gr.Image(
                                    value=path,
                                    interactive=False,
                                    show_label=False,
                                    height=110,
                                    width=110,
                                )
                                btn = gr.Button("识别", size="sm", variant="secondary")
                                btn.click(
                                    fn=lambda p=path, t=true_label: on_challenge(
                                        p, t
                                    ),
                                    outputs=[
                                        sketch,
                                        top_html,
                                        bars_html,
                                        msg,
                                        small_img,
                                        net_fig,
                                    ],
                                )

        sketch.input(
            fn=handle,
            inputs=sketch,
            outputs=[top_html, bars_html, msg, small_img, net_fig],
        )
        btn_random.click(random_example, outputs=sketch).then(
            fn=handle,
            inputs=sketch,
            outputs=[top_html, bars_html, msg, small_img, net_fig],
        )

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
