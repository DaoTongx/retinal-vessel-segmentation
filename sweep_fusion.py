# -*- coding: utf-8 -*-
"""Sweep fusion/postprocess variants on the 20 labelled train images."""
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
    p, g = pred.astype(np.float32), gt.astype(np.float32)
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


def remove_small(mask, ms):
    if ms <= 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= ms:
            out[lab == i] = 1
    return out


def fill_holes(mask):
    m = (mask.astype(np.uint8) * 255)
    h, w = m.shape
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    return ((m | cv2.bitwise_not(ff)) > 0).astype(np.uint8)


def make_variant(vd, ms):
    def fn(om, vm):
        if vd:
            k = np.ones((2 * vd + 1, 2 * vd + 1), np.uint8)
            vm = cv2.dilate(vm, k)
        m = ((om | vm) > 0).astype(np.uint8)
        return remove_small(m, ms)
    return fn


# ---------------- load ----------------
df = pd.read_csv(VCX / "bounds.csv")
d = np.load(OOF, allow_pickle=True)
probs, gts, names = d["probs"], d["gts"], list(d["names"])
ours_prob = {n: probs[i] for i, n in enumerate(names)}
gt_map = {n: gts[i] for i, n in enumerate(names)}

vmasks = {}
for _, row in df.iterrows():
    iid = str(row["id"])
    p = VCX / "vessels" / f"{iid}.png"
    gt = cv2.imread(str(LBL / f"{iid}.jpg"), cv2.IMREAD_GRAYSCALE)
    if p.exists() and gt is not None:
        m = map_back(cv2.imread(str(p), cv2.IMREAD_UNCHANGED),
                     parse_bounds(row["bounds"]), gt.shape)
        vmasks[f"{iid}.jpg"] = cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)

keys = [k for k in ours_prob if k in vmasks]
print(f"paired samples: {len(keys)}")

# ---------------- sweep ----------------
rows = []
for thr in (0.40, 0.45, 0.50, 0.55, 0.60):
    for vd in (0, 1, 2):
        for ms in (0, 20, 40, 80):
            fn = make_variant(vd, ms)
            ds = [dice(fn((ours_prob[k] > thr).astype(np.uint8), vmasks[k]), gt_map[k])
                  for k in keys]
            rows.append((f"thr{thr:.2f}_vdil{vd}_ms{ms}", float(np.mean(ds))))

# extra morphology on the plain union
for name, fn in (("union_fillholes", lambda om, vm: fill_holes((om | vm) > 0)),
                 ("union_close3", lambda om, vm: cv2.morphologyEx(
                     ((om | vm) > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                     np.ones((3, 3), np.uint8)))):
    ds = [dice(fn((ours_prob[k] > 0.50).astype(np.uint8), vmasks[k]), gt_map[k]) for k in keys]
    rows.append((name, float(np.mean(ds))))

rows.sort(key=lambda r: -r[1])
print()
print(f"{'variant':<26} {'dice':>8}")
print("-" * 36)
for n, v in rows[:16]:
    print(f"{n:<26} {v:>8.4f}")

base = [v for n, v in rows if n == "thr0.50_vdil0_ms0"][0]
print("-" * 36)
print(f"baseline (thr0.50, vdil0, ms0) = {base:.4f}")
print(f"best     = {rows[0][1]:.4f}  ({rows[0][0]})   gain {rows[0][1] - base:+.4f}")
