# -*- coding: utf-8 -*-
"""Extract training curves from the real log and generate all report figures."""
import re
import shutil
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(r"E:\machine learing\project4")
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LOG = ROOT / "run" / "logs" / "train_20261103_135243.log"
DATA = ROOT / "data" / "segmentation"


def setup_font():
    from matplotlib import font_manager
    want = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((w for w in want if w in have), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# ---------------- parse log ----------------
def parse_log():
    folds = {}
    cur = None
    pat = re.compile(r"\[fold (\d+)\] ep (\d+) \| loss ([\d.]+) \| dice ([\d.]+) \| best ([\d.]+)")
    for line in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            f, ep, loss, dice, best = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            folds.setdefault(f, []).append((ep, loss, dice, best))
    return folds


folds = parse_log()
print("folds parsed:", {k: len(v) for k, v in folds.items()})

BEST = {0: 0.7823, 1: 0.7492, 2: 0.7762, 3: 0.7599, 4: 0.7805}
BEST_EP = {0: 61, 1: 47, 2: 50, 3: 50, 4: 50}
STOP_EP = {0: 80, 1: 72, 2: 75, 3: 75, 4: 75}


def fig_curve():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    colors = ["#2f6fb3", "#d1682c", "#3f8f4f", "#9b59b6", "#7f8c8d"]

    for i, (f, rows) in enumerate(sorted(folds.items())):
        ep = [r[0] for r in rows]
        loss = [r[1] for r in rows]
        dice = [r[3] for r in rows]  # running best
        axes[0].plot(ep, loss, "-o", ms=2.6, lw=1.4, color=colors[i], label=f"fold {f}")
        axes[1].plot(ep, dice, "-o", ms=2.6, lw=1.4, color=colors[i], label=f"fold {f}")

    axes[0].set_xlabel("训练轮次 (epoch)")
    axes[0].set_ylabel("训练损失")
    axes[0].set_title("(a) 五折训练损失曲线", fontsize=10.5)
    axes[1].set_xlabel("训练轮次 (epoch)")
    axes[1].set_ylabel("验证集 Dice（历史最优）")
    axes[1].set_title("(b) 五折验证 Dice 曲线", fontsize=10.5)
    for ax in axes:
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=7.6, ncol=2)

    fig.tight_layout()
    p = FIG / "fig4-3_curve.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_folds():
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ks = sorted(BEST)
    vals = [BEST[k] * 100 for k in ks]
    bars = ax.bar([f"fold {k}" for k in ks], vals, color="#2f6fb3",
                  edgecolor="white", lw=0.6, width=0.58)
    for b, k in zip(bars, ks):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{BEST[k]:.4f}",
                ha="center", fontsize=8.6)
    mean = np.mean(vals)
    ax.axhline(mean, ls="--", lw=1.4, color="#c0392b", label=f"均值 {mean:.2f}%")
    ax.set_ylim(70, 81)
    ax.set_ylabel("最佳验证 Dice (%)")
    ax.set_title("五折交叉验证各折最佳 Dice（resnet34，512×512）", fontsize=10.5, pad=10)
    ax.legend(fontsize=8.4, loc="lower right")
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    p = FIG / "fig4-4_folds.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_data():
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
    for r, name in enumerate(["1.jpg", "13.jpg"]):
        rgb = cv2.cvtColor(cv2.imread(str(DATA / "train" / "image" / name)), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(DATA / "train" / "label" / name), cv2.IMREAD_GRAYSCALE)
        g = rgb[:, :, 1]

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"原始眼底图像 {name}（565×584）", fontsize=9.6)
        axes[r, 1].imshow(g, cmap="gray")
        axes[r, 1].set_title("绿通道（血管对比度最高）", fontsize=9.6)
        axes[r, 2].imshow(lbl, cmap="gray")
        axes[r, 2].set_title(f"标注掩膜（血管占比 {(lbl > 127).mean():.2%}）", fontsize=9.6)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    p = FIG / "fig4-1_data.png"
    fig.savefig(p, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fa", ec="#4a6fa5", fs=9.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.1, color="#5a6b7d"))

    box(2.6, 10.5, 4.8, 0.85, "训练集 20 张（565×584）+ 测试集 20 张", fc="#dbe7f5", bold=True)
    box(2.6, 9.35, 4.8, 0.8, "绿通道 CLAHE 增强 → 复制为 3 通道 → 重采样 512×512")
    arrow(5.0, 10.5, 5.0, 10.15)

    box(0.35, 7.95, 4.35, 0.95, "训练增强\n翻转 / ±35° 仿射 / 网格畸变\n亮度与伽马扰动")
    box(5.3, 7.95, 4.35, 0.95, "验证与推理\n仅重采样 + 标准化\n（不做随机增强）")
    arrow(5.0, 9.35, 2.5, 8.9)
    arrow(5.0, 9.35, 7.5, 8.9)

    box(0.35, 6.45, 9.3, 0.9,
        "U-Net 主干（segmentation-models-pytorch）+ ResNet34 编码器（ImageNet 预训练）\n输出单通道血管概率图",
        fs=8.9)
    arrow(2.5, 7.95, 3.2, 7.35)
    arrow(7.5, 7.95, 6.8, 7.35)

    box(2.6, 5.25, 4.8, 0.85, "损失：BCEWithLogits + Dice 组合\nAdamW + Warmup 5 轮 + 余弦退火", fc="#fdf1e3", ec="#c8873f", bold=True)
    arrow(5.0, 6.45, 5.0, 6.1)

    box(2.6, 3.95, 4.8, 0.85, "5 折交叉验证，每折早停（patience=25）\n输出 5 组权重与 OOF 概率图", fc="#fdf1e3", ec="#c8873f", bold=True)
    arrow(5.0, 5.25, 5.0, 4.8)

    box(0.35, 2.55, 4.35, 0.9, "OOF 扫描\n阈值 × 小连通域面积", fs=8.9)
    box(5.3, 2.55, 4.35, 0.9, "测试推理\n5 模型 × 8 路 TTA\n（4 旋转 × 2 翻转）", fs=8.9)
    arrow(4.2, 3.95, 2.5, 3.45)
    arrow(5.8, 3.95, 7.5, 3.45)

    box(2.6, 1.2, 4.8, 0.9, "与 VascX 预训练模型取并集（Union 融合）\n→ 512×512 二值掩膜（血管=255）", fc="#eaf5ea", ec="#4f8a4f", bold=True)
    arrow(2.5, 2.55, 4.2, 2.1)
    arrow(7.5, 2.55, 5.8, 2.1)

    box(2.6, 0.1, 4.8, 0.75, "调用官方 segmentation_to_csv.py → submission.csv（RLE）", fc="#f2eefa", ec="#6b5aa5", bold=True)
    arrow(5.0, 1.2, 5.0, 0.85)

    fig.tight_layout()
    p = FIG / "fig4-2_pipeline.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_fusion():
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    labels = ["U-Net\n(resnet34)", "VascX\n(预训练集成)", "并集融合\n(Union)"]
    vals = [0.7701, 0.7259, 0.7844]
    colors = ["#2f6fb3", "#7f8c8d", "#3f8f4f"]
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", lw=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", fontsize=9.4)
    ax.set_ylim(0.66, 0.81)
    ax.set_ylabel("训练集 20 张平均 Dice")
    ax.set_title("单一模型与并集融合的 Dice 对比（训练集 OOF 口径）", fontsize=10.5, pad=10)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    p = FIG / "fig4-7_fusion.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def fig_vascx_per_image():
    import pandas as pd

    vc = ROOT / "vascx_train_all"
    df = pd.read_csv(vc / "bounds.csv")
    d = np.load(ROOT / "run" / "oof.npz", allow_pickle=True)
    probs, gts, names = d["probs"], d["gts"], list(d["names"])

    def dice(p, g, eps=1e-6):
        p, g = p.astype(np.float32), g.astype(np.float32)
        return float((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))

    def parse(s):
        return eval(str(s), {"np": np, "numpy": np})

    def map_back(vimg, b, shape):
        hh, ww = shape
        cx, cy = float(b["center"][0]), float(b["center"][1])
        r = float(b["radius"])
        side = int(round(2 * r))
        sm = (vimg > 0).astype(np.uint8)
        if sm.shape[:2] != (side, side):
            sm = cv2.resize(sm, (side, side), interpolation=cv2.INTER_NEAREST)
        x0, y0 = int(round(cx - r)), int(round(cy - r))
        cv = np.zeros((hh, ww), np.uint8)
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(ww, x0 + side), min(hh, y0 + side)
        if dx1 > dx0 and dy1 > dy0:
            cv[dy0:dy1, dx0:dx1] = sm[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
        return cv

    ours, vx = [], []
    for _, row in df.iterrows():
        iid = str(row["id"])
        gt = cv2.imread(str(DATA / "train" / "label" / f"{iid}.jpg"), cv2.IMREAD_GRAYSCALE) > 127
        v = cv2.imread(str(vc / "vessels" / f"{iid}.png"), cv2.IMREAD_UNCHANGED)
        if v is None:
            continue
        m = map_back(v, parse(row["bounds"]), gt.shape)
        m512 = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)
        vx.append(dice(m512, gts[names.index(f"{iid}.jpg")]))
        ours.append(dice(probs[names.index(f"{iid}.jpg")] > 0.45, gts[names.index(f"{iid}.jpg")]))

    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    x = np.arange(len(ours))
    w = 0.38
    ax.bar(x - w / 2, np.array(ours) * 100, w, label="本实验 U-Net", color="#2f6fb3", edgecolor="white", lw=0.4)
    ax.bar(x + w / 2, np.array(vx) * 100, w, label="VascX", color="#8e9aa6", edgecolor="white", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(ours))], fontsize=7.6)
    ax.set_xlabel("训练集图像编号")
    ax.set_ylabel("Dice (%)")
    ax.set_ylim(40, 90)
    ax.set_title("逐图对比：本实验模型与 VascX 在 20 张标注图上的 Dice", fontsize=10.5, pad=8)
    ax.legend(fontsize=8.2, ncol=2, loc="lower left")
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    p = FIG / "fig4-6_vascx.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


if __name__ == "__main__":
    setup_font()
    print("fig data  :", fig_data())
    print("fig pipe  :", fig_pipeline())
    print("fig curve :", fig_curve())
    print("fig folds :", fig_folds())
    print("fig fusion:", fig_fusion())
    print("fig vascx :", fig_vascx_per_image())

    # reuse existing qualitative figures
    for src, dst in [("prediction_preview.png", "fig4-5_pred.png"),
                     ("vascx_vs_gt.png", "fig4-8_vascx_gt.png")]:
        s = ROOT / src
        if s.exists():
            shutil.copy(s, FIG / dst)
            print("copied:", dst)
