# -*- coding: utf-8 -*-
"""Evaluate VascX output against the DRIVE label, and render an overlay."""
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"E:\machine learing\project4")
OUT = ROOT / "vascx_single" / "out"
LBL_DIR = ROOT / "data" / "segmentation" / "train" / "label"
IMG_DIR = ROOT / "data" / "segmentation" / "train" / "image"


def dice(pred, gt, eps=1e-6):
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)
    inter = (p * g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


df = pd.read_csv(OUT / "bounds.csv")
print("bounds columns:", list(df.columns))
print(df.to_string())

v = cv2.imread(str(OUT / "vessels" / "1.png"), cv2.IMREAD_UNCHANGED)
print()
print("vessel png:", v.shape, v.dtype, "unique:", np.unique(v)[:10])

gt = cv2.imread(str(LBL_DIR / "1.jpg"), cv2.IMREAD_GRAYSCALE) > 127
print("label:", gt.shape, "vessel pixels:", int(gt.sum()))

def parse_bounds(s):
    """bounds 列里含 np.float64(...)，需要 eval 并注入 numpy。"""
    return eval(str(s), {"np": np, "numpy": np})


row = df.iloc[0]
b = parse_bounds(row["bounds"])
h, w = b["hw"]
cx, cy = float(b["center"][0]), float(b["center"][1])
R = float(b["radius"])
print(f"\nbounds: hw=({h},{w}) center=({cx:.1f},{cy:.1f}) R={R:.1f}")


def map_back(vessel_img, bounds, shape):
    """把 1024(或 2R) 空间的血管图按 center/radius 贴回原图。"""
    hh, ww = shape
    cxx, cyy = float(bounds["center"][0]), float(bounds["center"][1])
    rr = float(bounds["radius"])
    side = int(round(2 * rr))

    small = (vessel_img > 0).astype(np.uint8)
    if small.shape[:2] != (side, side):
        small = cv2.resize(small, (side, side), interpolation=cv2.INTER_NEAREST)

    x0 = int(round(cxx - rr))
    y0 = int(round(cyy - rr))
    # paste with clipping
    canvas = np.zeros((hh, ww), np.uint8)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(ww, x0 + side), min(hh, y0 + side)
    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = small[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return canvas


mapped = map_back(v, b, gt.shape)
print("\n=== scores on train image 1 ===")
d_all = dice(mapped, gt.astype(np.uint8))
print(f"VascX dice (full image)      : {d_all:.4f}")

# FOV-restricted dice
gray = cv2.cvtColor(cv2.imread(str(IMG_DIR / "1.jpg")), cv2.COLOR_BGR2GRAY)
fov = (gray > 10).astype(np.uint8)
d_fov = dice(mapped * fov, gt.astype(np.uint8) * fov)
print(f"VascX dice (inside fov)      : {d_fov:.4f}")
print(f"pred vessel ratio            : {mapped.mean():.4f}")
print(f"gt   vessel ratio            : {gt.mean():.4f}")

# --- overlay ---
bgr = cv2.imread(str(IMG_DIR / "1.jpg"))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
axes[0].imshow(rgb); axes[0].set_title("train image 1")
axes[1].imshow(mapped, cmap="gray"); axes[1].set_title("VascX vessel (mapped)")
axes[2].imshow(gt, cmap="gray"); axes[2].set_title("ground truth")
axes[3].imshow(rgb)
axes[3].imshow(np.ma.masked_where(gt == 0, gt), cmap="Greens", alpha=0.55)
axes[3].imshow(np.ma.masked_where(mapped == 0, mapped), cmap="Reds", alpha=0.45)
axes[3].set_title("green=GT  red=VascX")
for a in axes:
    a.axis("off")
plt.tight_layout()
p = ROOT / "vascx_vs_gt.png"
plt.savefig(p, dpi=110, bbox_inches="tight")
print("\nsaved:", p)
