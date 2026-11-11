# -*- coding: utf-8 -*-
"""Verify v2-specific logic: TTA inverse transforms, postprocess, threshold sweep."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def remove_small_components(mask_uint8, min_size):
    if min_size <= 0:
        return mask_uint8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num <= 1:
        return mask_uint8
    keep = np.zeros_like(mask_uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            keep[labels == i] = 1
    return keep


def dice_from_mask(pred_bin, target_bin, eps=1e-6):
    p = pred_bin.astype(np.float32)
    t = target_bin.astype(np.float32)
    inter = (p * t).sum()
    return (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def test_tta_inverse():
    """A rot/flip transform followed by its inverse must return the input."""
    torch.manual_seed(0)
    x0 = torch.randn(1, 3, 64, 64)

    ok = True
    for k in range(4):
        for flip in (False, True):
            x1 = torch.rot90(x0, k, dims=[2, 3])
            x2 = torch.flip(x1, dims=[3]) if flip else x1

            # simulate a model that just returns its input (batch, 1, H, W)
            p = x2[:, :1]
            if flip:
                p = torch.flip(p, dims=[3])
            p = torch.rot90(p, -k, dims=[2, 3])

            same = torch.allclose(p, x0[:, :1], atol=1e-6)
            ok = ok and same
            print(f"  rot={k * 90:3d} flip={flip!s:5s} -> inverse exact: {same}")
    return ok


def test_multiscale_shapes():
    """Scale -> size must stay a multiple of 32 and be resizable back to 512."""
    IMG = 512
    ok = True
    for scale in (0.75, 1.0, 1.25):
        s = int(round(IMG * scale / 32.0)) * 32
        good = (s % 32 == 0) and (s >= 64)
        ok = ok and good
        print(f"  scale={scale} -> {s} (mod32={s % 32})")
    # interpolate round trip
    x = torch.randn(1, 1, 640, 640)
    y = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    print(f"  interpolate 640->512: {tuple(y.shape)}")
    return ok and tuple(y.shape) == (1, 1, 512, 512)


def test_postprocess():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:12] = 1      # area 40, long vessel-like
    mask[50, 50] = 1            # area 1, noise
    mask[55:57, 55:57] = 1      # area 4, noise

    out0 = remove_small_components(mask, 0)
    out10 = remove_small_components(mask, 10)
    out20 = remove_small_components(mask, 20)

    print(f"  min_size=0  -> pixels {int(out0.sum())} (expect {int(mask.sum())})")
    print(f"  min_size=10 -> pixels {int(out10.sum())} (expect 40)")
    print(f"  min_size=20 -> pixels {int(out20.sum())} (expect 40)")

    ok = (out0.sum() == mask.sum()) and (out10.sum() == 40) and (out20.sum() == 40)
    return bool(ok)


def test_threshold_sweep():
    """Synthetic OOF: a perfect prob map plus a noisy one, sweep should recover it."""
    rng = np.random.default_rng(0)
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 15:17] = 1

    probs = {}
    masks = {}
    # sample A: clean
    probs["a"] = np.clip(gt.astype(np.float32) * 0.95 + rng.normal(0, 0.03, gt.shape), 0, 1)
    masks["a"] = gt.astype(np.float32)
    # sample B: shifted/thicker
    gt2 = np.zeros_like(gt)
    gt2[10:22, 14:19] = 1
    probs["b"] = np.clip(gt2.astype(np.float32) * 0.9 + rng.normal(0, 0.05, gt.shape), 0, 1)
    masks["b"] = gt2.astype(np.float32)

    grid = []
    for thr in np.round(np.arange(0.30, 0.76, 0.05), 2):
        for min_size in [0, 10, 30]:
            ds = []
            for n in probs:
                m = (probs[n] > thr).astype(np.uint8)
                m = remove_small_components(m, min_size)
                ds.append(dice_from_mask(m, masks[n]))
            grid.append((thr, min_size, float(np.mean(ds))))

    grid.sort(key=lambda x: -x[2])
    print(f"  best: thr={grid[0][0]:.2f} min_size={grid[0][1]} dice={grid[0][2]:.4f}")
    return grid[0][2] > 0.9


if __name__ == "__main__":
    print("[1] TTA inverse transforms")
    r1 = test_tta_inverse()
    print("\n[2] multi-scale sizes")
    r2 = test_multiscale_shapes()
    print("\n[3] connected-component filtering")
    r3 = test_postprocess()
    print("\n[4] threshold sweep on synthetic OOF")
    r4 = test_threshold_sweep()

    print("\n" + "=" * 50)
    print("ALL PASS:", all([r1, r2, r3, r4]))
