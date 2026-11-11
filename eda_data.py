# -*- coding: utf-8 -*-
"""Quick EDA of the fundus vessel segmentation dataset (local check)."""
import os
import numpy as np
from PIL import Image

ROOT = r"E:\machine learing\project4\data\segmentation"


def load_ids(folder):
    names = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]
    return sorted(names, key=lambda x: int(os.path.splitext(x)[0]))


for split in ["train/image", "train/label", "test/image"]:
    d = os.path.join(ROOT, split)
    ids = load_ids(d)
    sizes, modes = set(), set()
    for f in ids:
        im = Image.open(os.path.join(d, f))
        sizes.add(im.size)
        modes.add(im.mode)
    print(f"[{split}] n={len(ids)} sizes={sorted(sizes)} modes={sorted(modes)}")

print()
print("=== label pixel statistics ===")
label_dir = os.path.join(ROOT, "train", "label")
for f in load_ids(label_dir)[:6]:
    arr = np.array(Image.open(os.path.join(label_dir, f)))
    uniq, cnt = np.unique(arr, return_counts=True)
    ratio = cnt / arr.size
    info = ", ".join(f"{u}->{r:.4f}" for u, r in zip(uniq, ratio))
    print(f"{f:>8} shape={arr.shape} dtype={arr.dtype} values: {info}")

print()
print("=== image pixel statistics ===")
img_dir = os.path.join(ROOT, "train", "image")
for f in load_ids(img_dir)[:3]:
    arr = np.array(Image.open(os.path.join(img_dir, f)).convert("L"))
    print(f"{f:>8} shape={arr.shape} min={arr.min()} max={arr.max()} mean={arr.mean():.2f}")
