# -*- coding: utf-8 -*-
"""
Build the final submission: union of our UNet prediction and VascX.

On the 20 labelled train images:
    ours 0.7696 | vascx 0.7259 | union 0.7844
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(r"E:\machine learing\project4")
OURS = ROOT / "run" / "predictions"
VCX = ROOT / "vascx_test_all"
OUT = ROOT / "union_run"
DATA = ROOT / "data" / "segmentation"


def parse_bounds(s):
    return eval(str(s), {"np": np, "numpy": np})


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


def remove_small_components(mask, min_size):
    if min_size <= 0:
        return mask
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[lab == i] = 1
    return out


bounds = pd.read_csv(VCX / "bounds.csv")
bmap = {str(row["id"]): parse_bounds(row["bounds"]) for _, row in bounds.iterrows()}
print(f"bounds rows: {len(bmap)}")

if OUT.exists():
    shutil.rmtree(OUT)
pred = OUT / "image"
pred.mkdir(parents=True)

ids = sorted([p.stem for p in (DATA / "test" / "image").glob("*.jpg")], key=int)
print(f"test ids: {len(ids)}")

rows = []
for iid in ids:
    ours_p = OURS / f"{iid}.png"
    vcx_p = VCX / "vessels" / f"{iid}.png"
    if not ours_p.exists() or not vcx_p.exists():
        print(f"  {iid}: MISSING (ours={ours_p.exists()}, vascx={vcx_p.exists()})")
        continue

    ours = (cv2.imread(str(ours_p), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    v = cv2.imread(str(vcx_p), cv2.IMREAD_UNCHANGED)
    gt_shape = (584, 565)
    vmap = map_back(v, bmap[iid], gt_shape)
    v512 = cv2.resize(vmap, (512, 512), interpolation=cv2.INTER_NEAREST)

    u = (ours | v512).astype(np.uint8)
    u = remove_small_components(u, 20)

    cv2.imwrite(str(pred / f"{iid}.png"), u * 255)
    rows.append((int(iid), float(ours.mean()), float(v512.mean()), float(u.mean())))

print()
print(f"{'id':>4} {'ours':>8} {'vascx':>8} {'union':>8}")
for iid, a, b, c in rows:
    print(f"{iid:>4} {a:>8.4f} {b:>8.4f} {c:>8.4f}")
print()
print(f"mean ratios: ours={np.mean([r[1] for r in rows]):.4f} "
      f"vascx={np.mean([r[2] for r in rows]):.4f} "
      f"union={np.mean([r[3] for r in rows]):.4f}")

# official csv
shutil.copy(str(DATA / "segmentation_to_csv.py"), str(OUT / "segmentation_to_csv.py"))
res = subprocess.run([sys.executable, "segmentation_to_csv.py"], cwd=str(OUT),
                     capture_output=True, text=True)
print("official script rc:", res.returncode)
if res.returncode != 0:
    print(res.stderr[-1500:])

sub = OUT / "submission.csv"
df = pd.read_csv(sub)
df["Id"] = df["Id"].astype(int)
df = df.sort_values("Id").reset_index(drop=True)
df.to_csv(sub, index=False)
print(f"submission -> {sub} ({len(df)} rows)")

# sanity check
tot = []
for _, r in df.iterrows():
    vv = list(map(int, str(r["Predicted"]).split()))
    assert all(x >= 1 for x in vv)
    s = vv[0::2]
    assert all(s[i] < s[i + 1] for i in range(len(s) - 1))
    tot.append(sum(vv[1::2]) / (512 * 512))
print(f"pixel ratio: mean={np.mean(tot):.4f} min={np.min(tot):.4f} max={np.max(tot):.4f}")
