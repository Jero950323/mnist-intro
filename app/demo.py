"""手写数字识别演示（产品化页面）。

用法（在项目根目录）:
    python app/demo.py
    或
    python app/demo.py --share
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.model import build_model
from src.network_viz import build_cnn_html, build_empty_html, build_viewer_html
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


# ---------- 识别 ----------


def classify(model, img):
    """把画板输入转成 28x28 并预测，返回 (probs, canvas, tensor)。"""
    if img is None:
        return None, None, None
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


def render_bars(probs):
    """0~9 概率条形图（HTML）。"""
    order = np.argsort(probs)[::-1]
    items = []
    for rank, idx in enumerate(order):
        p = probs[idx]
        w = max(2.0, p * 100)
        top = rank == 0
        color = "#2563eb" if top else "#b6ccf7"
        label_color = "#1d4ed8" if top else "#64748b"
        items.append(
            f"""
            <div class="prob-row">
              <span class="prob-label" style="color:{label_color}">{idx}</span>
              <div class="prob-track">
                <div class="prob-fill" style="width:{w:.1f}%;background:{color}"></div>
              </div>
              <span class="prob-val" style="color:{label_color}">{p*100:.1f}%</span>
            </div>
            """
        )
    return "<div>" + "".join(items) + "</div>"


def render_top(top1, conf):
    """顶部大字结果 + 置信度徽章。"""
    level = "conf-high" if conf >= 0.9 else ("conf-mid" if conf >= 0.6 else "conf-low")
    return f"""
    <div class="result-hero">
      <div class="result-label">识别结果</div>
      <div class="result-digit">{top1}</div>
      <div class="result-conf">
        <span class="conf-badge {level}">置信度 {conf:.1%}</span>
      </div>
    </div>
    """


def predict(model, img):
    """把画板图像转成 28x28 并预测，返回页面组件需要的内容。"""
    empty_top = (
        "<div style='text-align:center;color:#94a3b8;font-size:18px;padding:30px 0;'>"
        "请先在画板上写一个数字</div>"
    )
    empty_bars = (
        "<div style='text-align:center;color:#94a3b8;padding:12px;'>"
        "写完之后，这里会显示 0~9 每个数字的概率</div>"
    )
    if img is None:
        return empty_top, empty_bars, "", None
    probs, canvas, _ = classify(model, img)
    if probs is None:
        return (
            "<div style='text-align:center;color:#f59e0b;font-size:16px;padding:30px 0;'>"
            "没有识别到笔画，请再写一次，或点击左侧「🔍 识别」按钮</div>",
            empty_bars,
            "",
            None,
        )
    top1 = int(probs.argmax())
    conf = float(probs[top1])
    msg = (
        f"识别结果：**{top1}**（置信度 {conf:.1%}）"
        "　提示：写大、写粗、写中间最准；"
        f"{'当前置信度较低，可以重写一次试试。' if conf < 0.4 else '左侧条形图是 0~9 每个数字的概率。'}"
    )
    return (
        render_top(top1, conf),
        render_bars(probs),
        msg,
        (canvas * 255).astype(np.uint8),
    )


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
    """创建手写画板组件，兼容不同版本的 Gradio。"""
    import inspect

    params = inspect.signature(gr.Sketchpad.__init__).parameters
    if "shape" in params:
        return gr.Sketchpad(
            shape=(420, 420),
            image_mode="L",
            invert_colors=True,
            brush_radius=26,
            label="",
        )
    return gr.Sketchpad(
        height=420,
        width=420,
        image_mode="L",
        canvas_size=(560, 560),
        brush=gr.Brush(default_size=26),
        label="",
    )


CSS = """
.gradio-container {
  background: linear-gradient(160deg, #eef2ff 0%, #f8fafc 45%, #f1f5f9 100%);
  max-width: 1200px !important;
  margin: 0 auto;
  font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
}

/* 品牌头部 */
.app-header {
  background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%);
  color: #fff;
  border-radius: 18px;
  padding: 24px 30px;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  box-shadow: 0 10px 30px rgba(37, 99, 235, 0.25);
}
.brand-name { font-size: 25px; font-weight: 800; letter-spacing: 1px; }
.brand-sub { font-size: 14px; opacity: .85; margin-top: 4px; }
.header-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
  background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.35);
  border-radius: 999px;
  padding: 6px 14px;
  font-size: 13px;
}

/* 卡片 */
.card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  box-shadow: 0 4px 18px rgba(15, 23, 42, .06);
  padding: 20px 22px;
}
.section-title {
  margin: 0 0 12px !important;
  font-size: 16px !important;
  color: #0f172a !important;
}
.tips {
  color: #64748b;
  font-size: 13px;
  line-height: 1.7;
  background: #f8fafc;
  border-radius: 10px;
  padding: 10px 14px;
  margin-top: 12px;
}

/* 结果区 */
.result-hero { text-align: center; padding: 8px 0 6px; }
.result-label { font-size: 13px; color: #94a3b8; letter-spacing: 2px; font-weight: 600; }
.result-digit { font-size: 84px; font-weight: 800; color: #1d4ed8; line-height: 1.15; }
.result-conf { margin-top: 8px; }
.conf-badge { display: inline-block; border-radius: 999px; padding: 5px 16px; font-size: 14px; font-weight: 600; }
.conf-high { background: #dcfce7; color: #15803d; }
.conf-mid  { background: #fef3c7; color: #b45309; }
.conf-low  { background: #fee2e2; color: #b91c1c; }

/* 概率条 */
.prob-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.prob-label { width: 22px; font-weight: 700; text-align: center; color: #475569; }
.prob-track { flex: 1; background: #eef2f7; border-radius: 8px; height: 22px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 8px; transition: width .18s ease; }
.prob-val { width: 56px; text-align: right; font-family: Consolas, monospace; font-size: 13px; color: #475569; }

/* 页脚 */
.app-footer { text-align: center; color: #94a3b8; font-size: 13px; padding: 18px 0 8px; }
footer { display: none !important; }
"""


def main():
    parser = argparse.ArgumentParser(description="MNIST 手写识别交互演示")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--checkpoint", default="outputs/cnn_mnist.pth")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    model = load_model(args.model, args.checkpoint)
    viz_model = model if args.model == "cnn" else load_model("cnn", "outputs/cnn_mnist.pth")

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

    # 记住最近一次识别，供“分步查看”按钮使用
    state = {"tensor": None, "link": ""}

    def write_viewer(tensor):
        """把当前数字的独立分步演示页写到文件，并生成“放大”链接。"""
        viewer_path = ROOT / "outputs" / "net_viewer.html"
        viewer_path.write_text(build_viewer_html(viz_model, tensor), encoding="utf-8")
        state["tensor"] = tensor
        state["link"] = (
            f'<a href="{viewer_path.as_uri()}" target="_blank" '
            'style="font-size:14px;color:#2563eb;font-weight:600;'
            'text-decoration:none;border:1px solid #bfdbfe;'
            'border-radius:10px;padding:6px 12px;display:inline-block;'
            'background:#eff6ff;">🔍 放大 · 分步演示（新窗口）</a>'
        )

    def handle(img):
        top, bars, msg, _ = predict(model, img)
        _, _, tensor = classify(model, img)
        if tensor is not None:
            viz = build_cnn_html(viz_model, tensor)
            write_viewer(tensor)
        else:
            viz = build_empty_html()
            state["tensor"] = None
            state["link"] = ""
        return top, bars, msg, viz, state["link"]

    def show_step(step):
        if state["tensor"] is None:
            return build_empty_html(), state["link"]
        return build_cnn_html(viz_model, state["tensor"], step=step), state["link"]

    def show_auto():
        if state["tensor"] is None:
            return build_empty_html(), state["link"]
        return build_cnn_html(viz_model, state["tensor"]), state["link"]

    def random_example():
        return gr.update(value=random.choice(example_paths))

    def on_challenge(path, true_label):
        """挑战题：识别后揭晓答案，讲解模型的盲区。"""
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        probs, _, tensor = classify(model, img)
        if probs is None:
            return (
                gr.update(value=path),
                empty_top,
                empty_bars,
                "",
                build_empty_html(),
                state["link"],
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
        write_viewer(tensor)
        return (
            gr.update(value=path),
            render_top(top1, conf),
            render_bars(probs),
            reveal,
            build_cnn_html(viz_model, tensor),
            state["link"],
        )

    gradio_major = int(gr.__version__.split(".")[0])
    blocks_kwargs = {"title": "手写数字识别 · 深度学习原理演示"}
    launch_kwargs = {"share": args.share}
    if gradio_major >= 6:
        launch_kwargs.update(css=CSS, theme=gr.themes.Soft(primary_hue="blue"))
    else:
        blocks_kwargs.update(css=CSS, theme=gr.themes.Soft(primary_hue="blue"))

    with gr.Blocks(**blocks_kwargs) as demo:
        # 品牌头部
        gr.HTML(
            """
            <div class="app-header">
              <div>
                <div class="brand-name">✏️ 手写数字识别</div>
                <div class="brand-sub">用大白话看懂深度学习 · PyTorch 教学演示</div>
              </div>
              <div class="header-badges">
                <span class="badge">CNN 准确率 98.95%</span>
                <span class="badge">本地运行 · 数据不上传</span>
                <span class="badge">识别过程可视化</span>
              </div>
            </div>
            """
        )

        # 主操作区：左画板 + 右结果
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes=["card"]):
                gr.Markdown("### ✍️ 手写输入", elem_classes=["section-title"])
                sketch = build_sketchpad()
                with gr.Row():
                    btn_random = gr.Button("🎲 随机示例", variant="secondary")
                    btn_go = gr.Button("🔍 识别", variant="primary")
                gr.Markdown(
                    "**提示**：写大、写粗、写中间最准；“1”直接画一根竖线。",
                    elem_classes=["tips"],
                )

            with gr.Column(scale=4):
                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("### 📊 识别结果", elem_classes=["section-title"])
                    top_html = gr.HTML(empty_top)
                    bars_html = gr.HTML(empty_bars)
                    msg = gr.Markdown("")

        # 计算过程动画
        with gr.Column(elem_classes=["card"]):
            with gr.Row():
                gr.Markdown(
                    "### 🧠 CNN 是怎么算出来的（小白版动画）",
                    elem_classes=["section-title"],
                )
                viewer_link = gr.HTML("")
            gr.Markdown(
                "完整流程：**看 → 找模板 → 组合 → 压缩 → 投票打分 → 出结果**。"
                "输入 28×28 → 卷积层 1（32 张特征图）→ 卷积层 2（64 张）→ 池化"
                "（64 张 7×7）→ **全连接层给 0~9 各打一个分**（正分支持、负分反对）"
                "→ 输出概率。最下面的“为什么判断是 X”证据图，用高亮标出模型重点看的区域。"
                "也可以点下面的步骤按钮，一步一步看。"
            )
            net_html = gr.HTML(build_empty_html())
            with gr.Row():
                step_labels = [
                    "① 输入",
                    "② 找模板",
                    "③ 组合",
                    "④ 压缩",
                    "⑤ 投票",
                    "⑥ 结果",
                    "⑦ 证据",
                ]
                for idx, label in enumerate(step_labels, 1):
                    gr.Button(label, size="sm", variant="secondary").click(
                        fn=lambda s=idx: show_step(s),
                        outputs=[net_html, viewer_link],
                    )
                gr.Button("▶ 自动演示", size="sm", variant="primary").click(
                    fn=show_auto,
                    outputs=[net_html, viewer_link],
                )

        # 样本库：示例 + 挑战题
        with gr.Tabs():
            with gr.Tab("📁 示例（0~9）"):
                gr.Examples(
                    examples=[[p] for p in example_paths],
                    inputs=sketch,
                    label="点一张，自动填入画板并识别",
                )
            if challenges:
                with gr.Tab("🧩 挑战题（模型的盲区）"):
                    gr.Markdown(
                        "先猜猜这些数字是几，再点「识别」。模型答错的题，最能说明"
                        "**训练数据长什么样，模型就学什么样**。"
                    )
                    for i in range(0, len(challenges), 4):
                        with gr.Row():
                            for path, true_label in challenges[i : i + 4]:
                                with gr.Column(min_width=110):
                                    gr.Image(
                                        value=path,
                                        interactive=False,
                                        show_label=False,
                                        height=110,
                                        width=110,
                                    )
                                    btn = gr.Button(
                                        "识别", size="sm", variant="secondary"
                                    )
                                    btn.click(
                                        fn=lambda p=path, t=true_label: on_challenge(
                                            p, t
                                        ),
                                        outputs=[
                                            sketch,
                                            top_html,
                                            bars_html,
                                            msg,
                                            net_html,
                                            viewer_link,
                                        ],
                                    )

        # 页脚
        gr.HTML(
            '<div class="app-footer">PyTorch · MNIST 教学项目 · '
            "手写数字识别 —— 用大白话看懂深度学习</div>"
        )

        # 事件
        sketch.input(
            fn=handle,
            inputs=sketch,
            outputs=[top_html, bars_html, msg, net_html, viewer_link],
        )
        btn_random.click(random_example, outputs=sketch).then(
            fn=handle,
            inputs=sketch,
            outputs=[top_html, bars_html, msg, net_html, viewer_link],
        )
        btn_go.click(
            fn=handle,
            inputs=sketch,
            outputs=[top_html, bars_html, msg, net_html, viewer_link],
        )

    # 预启动处理队列 + 预热，避免第一次手写"没反应"
    demo.queue(default_concurrency_limit=4)
    try:
        if example_paths:
            warm = np.asarray(
                Image.open(example_paths[0]).convert("L"), dtype=np.float32
            )
            handle(warm)
            print("预热完成：模型、特征图与页面组件已就绪")
    except Exception as exc:
        print(f"预热未完成（不影响使用）: {exc}")

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
