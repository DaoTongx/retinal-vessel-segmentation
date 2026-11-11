# -*- coding: utf-8 -*-
"""Verify the official RLE semantics against our notebook implementation."""
import importlib.util
import os
import sys

import numpy as np
from PIL import Image

ROOT = r"E:\machine learing\project4\data\segmentation"
OFFICIAL = os.path.join(ROOT, "segmentation_to_csv.py")


def load_official():
    spec = importlib.util.spec_from_file_location("seg2csv", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rle_encode(mask_2d):
    flat = mask_2d.flatten()
    dots = np.where(flat > 0)[0]
    out = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            out.extend((b + 1, 0))
        out[-1] += 1
        prev = b
    return " ".join(str(x) for x in out)


def rle_decode(rle_str, h=512, w=512):
    mask = np.zeros(h * w, dtype=np.uint8)
    if rle_str.strip():
        vals = list(map(int, rle_str.split()))
        for i in range(0, len(vals), 2):
            s, l = vals[i], vals[i + 1]
            mask[s - 1: s - 1 + l] = 1
    return mask.reshape(h, w)


def main():
    official = load_official()

    # --- A. real label -> copy to 512x512, vessel=255 ---
    lbl = np.array(Image.open(os.path.join(ROOT, "train", "label", "1.jpg")).convert("L"))
    print("label shape:", lbl.shape, "vessel ratio:", (lbl > 127).mean())

    vessel_mask = ((lbl > 127) * 255).astype(np.uint8)
    img = Image.fromarray(vessel_mask).resize((512, 512), Image.Resampling.NEAREST)
    tmp = os.path.join(os.environ.get("TEMP", "."), "_notebook_test_512.png")
    img.save(tmp)

    official_rle = official.turn_to_str([tmp])[0]
    ours = rle_encode(np.array(img))
    print("\n[A] 512x512 input, vessel=255")
    print("  official == ours:", official_rle == ours)
    print("  non-empty:", len(ours) > 0, "| pairs:", len(ours.split()) // 2)

    # round trip
    dec = rle_decode(official_rle)
    orig = (np.array(img) > 0).astype(np.uint8)
    print("  round-trip identical:", np.array_equal(dec, orig))

    # 1-based / sorted / no duplicate check
    vals = list(map(int, official_rle.split()))
    starts = vals[0::2]
    ok_sorted = all(starts[i] < starts[i + 1] for i in range(len(starts) - 1))
    ok_positive = all(v >= 1 for v in vals)
    print("  starts strictly increasing:", ok_sorted, "| all values >= 1:", ok_positive)

    # --- B. inverted convention: vessel=0, background=255 ---
    inverted = np.where(vessel_mask > 0, 0, 255).astype(np.uint8)
    img2 = Image.fromarray(inverted).resize((512, 512), Image.Resampling.NEAREST)
    tmp2 = os.path.join(os.environ.get("TEMP", "."), "_notebook_test_inverted.png")
    img2.save(tmp2)
    inv_rle = official.turn_to_str([tmp2])[0]
    inv_dec = rle_decode(inv_rle)
    print("\n[B] inverted convention (vessel=0, bg=255)")
    print("  decoded == background:", np.array_equal(inv_dec, 1 - orig))
    print("  decoded == vessel   :", np.array_equal(inv_dec, orig))
    print("  -> confirms: official script encodes NON-ZERO pixels only")

    # --- C. non-512 input: does BILINEAR resize inflate the mask? ---
    small = Image.fromarray(vessel_mask).resize((400, 400), Image.Resampling.NEAREST)
    tmp3 = os.path.join(os.environ.get("TEMP", "."), "_notebook_test_400.png")
    small.save(tmp3)
    small_rle = official.turn_to_str([tmp3])[0]
    small_dec = rle_decode(small_rle)
    n_src = (np.array(small.resize((512, 512), Image.Resampling.NEAREST)) > 0).sum()
    print("\n[C] non-512 input (400x400 source)")
    print("  pixels encoded:", int(small_dec.sum()))
    print("  nearest-resize pixels:", int(n_src))
    print("  inflation:", int(small_dec.sum()) - int(n_src))

    for f in (tmp, tmp2, tmp3):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    sys.exit(main())
