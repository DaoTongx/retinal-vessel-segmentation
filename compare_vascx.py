# -*- coding: utf-8 -*-
"""Compare VascX vs our UNet on all 20 labelled train images."""
import numpy as np
import cv2
import pandas as pd
from pathlib import Path

ROOT = Path(r"E:\machine learing\project4")
VCX = ROOT / "vascx_train_all"
LBL = ROOT / "data" / "segmentation" / "train" / "label"
OOF = ROOT / "run" / "oof.npz"


def parse_bounds(s):
    return eval(str(s), {"np": np, "numpy": np})


def dice(pred, gt, eps=1e-6):
    p = pred.astype(np.float32)
    g = gt.astype(np.float32)
    inter = (p * g).sum()
    return float((2 * inter + eps) / (p.sum() + g.sum() + eps))


def map_back(vimg, b, shape):
    hh, ww = shape
    cx, cy = float(b["center"][0]), float(b["center"][1])
    r = float(b["radius"])
    side = int(round(2 * r))

    small = (vimg > 0).astype(np.uint8)
    if small.shape[:2] != (side, side):
        small = cv2.resize(small, (side, side), interpolation=cv2.INTER_NEAREST)

    x0, y0 = int(round(cx - r)), int(round(cy - r))
    canvas = np.zeros((hh, ww), np.uint8)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(ww, x0 + side), min(hh, y0 + side)
    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = small[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return canvas


df = pd.read_csv(VCX / "bounds.csv")
print(f"bounds rows: {len(df)}")

d_oof = np.load(OOF, allow_pickle=True)
probs, gts, names = d_oof["probs"], d_oof["gts"], list(d_oof["names"])
ours = {n: dice(probs[i] > 0.45, gts[i]) for i, n in enumerate(names)}
print(f"our OOF samples: {len(ours)}")

rows = []
for _, row in df.iterrows():
    iid = str(row["id"])
    vpath = VCX / "vessels" / f"{iid}.png"
    gt_path = LBL / f"{iid}.jpg"
    if not vpath.exists() or not gt_path.exists():
        continue
    v = cv2.imread(str(vpath), cv2.IMREAD_UNCHANGED)
    b = parse_bounds(row["bounds"])
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127
    m = map_back(v, b, gt.shape)
    rows.append({
        "id": int(iid),
        "vascx": dice(m, gt.astype(np.uint8)),
        "ours": ours.get(f"{iid}.jpg", float("nan")),
        "vascx_ratio": float(m.mean()),
        "gt_ratio": float(gt.mean()),
    })

r = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
print()
print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print()
print("=" * 60)
print(f"VascX  mean dice : {r['vascx'].mean():.4f}  (std {r['vascx'].std():.4f})")
print(f"Ours   mean dice : {r['ours'].mean():.4f}  (std {r['ours'].std():.4f})")
print(f"Delta            : {(r['vascx'] - r['ours']).mean():+.4f}")
print("=" * 60)
print(f"VascX ratio mean : {r['vascx_ratio'].mean():.4f}")
print(f"GT    ratio mean : {r['gt_ratio'].mean():.4f}")
