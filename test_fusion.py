# -*- coding: utf-8 -*-
"""Can we fuse VascX with our UNet? Test union / dilation on 20 train images."""
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


df = pd.read_csv(VCX / "bounds.csv")
d = np.load(OOF, allow_pickle=True)
probs, gts, names = d["probs"], d["gts"], list(d["names"])
ours_prob = {n: probs[i] for i, n in enumerate(names)}
gt_map = {n: gts[i] for i, n in enumerate(names)}

# VascX masks mapped back to 565x584
vmasks = {}
for _, row in df.iterrows():
    iid = str(row["id"])
    p = VCX / "vessels" / f"{iid}.png"
    gt = cv2.imread(str(LBL / f"{iid}.jpg"), cv2.IMREAD_GRAYSCALE)
    if p.exists() and gt is not None:
        vmasks[f"{iid}.jpg"] = map_back(cv2.imread(str(p), cv2.IMREAD_UNCHANGED),
                                        parse_bounds(row["bounds"]), gt.shape)

print(f"vascx masks: {len(vmasks)}, oof: {len(ours_prob)}")

# note: OOF probs are 512x512, vascx masks are 565x584 -> resize vascx to 512
def to512(m):
    return cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST)


results = {}
for thr in (0.40, 0.45, 0.50):
    a, b, u, i, vd3, vd5 = [], [], [], [], [], []
    for n in ours_prob:
        if n not in vmasks:
            continue
        gt = gt_map[n]
        om = (ours_prob[n] > thr).astype(np.uint8)
        vm = to512(vmasks[n])
        a.append(dice(om, gt))
        b.append(dice(vm, gt))
        u.append(dice(om | vm, gt))
        i.append(dice(om & vm, gt))
        vd3.append(dice(cv2.dilate(vm, np.ones((3, 3), np.uint8)), gt))
        vd5.append(dice(cv2.dilate(vm, np.ones((5, 5), np.uint8)), gt))
    results[thr] = dict(ours=np.mean(a), vascx=np.mean(b), union=np.mean(u),
                        intersect=np.mean(i), vascx_d3=np.mean(vd3), vascx_d5=np.mean(vd5))

print()
print(f"{'thr':>5} {'ours':>8} {'vascx':>8} {'union':>8} {'inter':>8} {'vx_d3':>8} {'vx_d5':>8}")
for thr, r in results.items():
    print(f"{thr:>5.2f} {r['ours']:>8.4f} {r['vascx']:>8.4f} {r['union']:>8.4f} "
          f"{r['intersect']:>8.4f} {r['vascx_d3']:>8.4f} {r['vascx_d5']:>8.4f}")

print()
print("结论：union 是否超过纯 ours？",
      max(r["union"] - r["ours"] for r in results.values()) > 0.005)
