# -*- coding: utf-8 -*-
"""Evaluate FOV masking on OOF, and render a few prediction overlays."""
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_local import make_fov_mask, remove_small_components, dice_np

ROOT = Path(r"E:\machine learing\project4")
DATA = ROOT / "data" / "segmentation"
RUN = ROOT / "run"

d = np.load(RUN / "oof.npz", allow_pickle=True)
probs, gts, names = d["probs"], d["gts"], list(d["names"])
print("OOF samples:", len(names), "| prob shape:", probs[0].shape)

# --- FOV effect on OOF ------------------------------------------------
fovs = {}
for n in names:
    bgr = cv2.imread(str(DATA / "train" / "image" / n))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fovs[n] = make_fov_mask(rgb, 512)

print()
print(f"{'thr':>5} {'fov':>5} {'dice':>8}  delta")
base = None
rows = []
for thr in (0.40, 0.45, 0.50):
    for use_fov in (False, True):
        ds = []
        for i, n in enumerate(names):
            m = (probs[i] > thr).astype(np.uint8)
            if use_fov:
                m = m * (fovs[n] > 0).astype(np.uint8)
            ds.append(dice_np(m, gts[i]))
        v = float(np.mean(ds))
        rows.append((thr, use_fov, v))
        if not use_fov and thr == 0.50:
            base = v
        print(f"{thr:>5.2f} {str(use_fov):>5} {v:>8.4f}")

print()
for thr in (0.40, 0.45, 0.50):
    a = [r[2] for r in rows if r[0] == thr and not r[1]][0]
    b = [r[2] for r in rows if r[0] == thr and r[1]][0]
    print(f"thr={thr:.2f}: FOV gain = {b-a:+.4f}")

# --- how much of the FOV mask actually cuts away ----------------------
frac = [float((fovs[n] > 0).mean()) for n in names]
print()
print(f"FOV area fraction: mean={np.mean(frac):.4f} (a circle in a square is ~0.785)")

# expected area outside FOV where model may fire
outside = []
for i, n in enumerate(names):
    m = (probs[i] > 0.45).astype(np.uint8)
    outside.append(float((m * (fovs[n] == 0)).sum()))
print(f"predicted vessel pixels OUTSIDE fov (thr .45): mean={np.mean(outside):.1f}, "
      f"max={np.max(outside):.0f}")

# --- overlay render for two test images -------------------------------
pred_dir = RUN / "predictions"
picks = ["1.png", "13.png"]
fig, axes = plt.subplots(len(picks), 3, figsize=(14, 4.5 * len(picks)))
if len(picks) == 1:
    axes = axes.reshape(1, -1)

for r, pf in enumerate(picks):
    stem = os.path.splitext(pf)[0]
    bgr = cv2.imread(str(DATA / "test" / "image" / f"{stem}.jpg"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(pred_dir / pf), cv2.IMREAD_GRAYSCALE)
    fov = make_fov_mask(rgb, 512)

    axes[r, 0].imshow(rgb)
    axes[r, 0].set_title(f"test image {stem}")
    axes[r, 1].imshow(mask, cmap="gray")
    axes[r, 1].set_title(f"prediction (ratio {(mask > 0).mean():.4f})")
    axes[r, 2].imshow(rgb)
    axes[r, 2].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.75)
    axes[r, 2].set_title("overlay")
    for ax in axes[r]:
        ax.axis("off")

plt.tight_layout()
out_png = ROOT / "prediction_preview.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print()
print("saved:", out_png)
