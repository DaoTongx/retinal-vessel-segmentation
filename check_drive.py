# -*- coding: utf-8 -*-
"""Check whether the dataset looks like DRIVE (FOV shape, EXIF, borders)."""
import os

import numpy as np
import cv2
from PIL import Image

ROOT = r"E:\machine learing\project4\data\segmentation"
IMG = os.path.join(ROOT, "train", "image")
LBL = os.path.join(ROOT, "train", "label")

for name in ["1.jpg", "5.jpg", "20.jpg"]:
    p = os.path.join(IMG, name)
    im = Image.open(p)
    exif = im.getexif()
    print(f"--- {name} size={im.size} mode={im.mode} exif_len={len(exif)}")
    if exif:
        for k, v in list(exif.items())[:10]:
            print(f"    exif {k}: {str(v)[:60]}")

    rgb = np.array(im.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # FOV: bright region
    fov = (gray > 20).astype(np.uint8)
    fov_ratio = fov.mean()

    # corners black?
    c = 20
    corners = {
        "TL": gray[:c, :c].mean(),
        "TR": gray[:c, -c:].mean(),
        "BL": gray[-c:, :c].mean(),
        "BR": gray[-c:, -c:].mean(),
    }

    # circularity: distance of fov boundary from center
    ys, xs = np.where(fov > 0)
    cy, cx = ys.mean(), xs.mean()
    radii = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    print(f"    gray mean={gray.mean():.1f} fov_ratio={fov_ratio:.3f} "
          f"center=({cy:.0f},{cx:.0f}) radius mean={radii.mean():.1f} std={radii.std():.1f}")
    print(f"    corners: {', '.join(f'{k}={v:.0f}' for k, v in corners.items())}")

    # vessel pixels inside vs outside FOV
    lbl = cv2.imread(os.path.join(LBL, name), cv2.IMREAD_GRAYSCALE)
    vessel = lbl > 127
    inside = vessel & (fov > 0)
    outside = vessel & (fov == 0)
    print(f"    vessel={vessel.sum()}  inside_fov={inside.sum()}  outside_fov={outside.sum()}")

print("\nDRIVE 特征对照: 图像 565x584, 圆形 FOV, 四角纯黑, 血管仅在 FOV 内")
