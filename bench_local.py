# -*- coding: utf-8 -*-
"""Local benchmark: albumentations API compatibility + CPU training speed."""
import time

import numpy as np
import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

print("=== albumentations", A.__version__, "===")
try:
    t = A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=35, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    print("Compose build: OK")
    img = np.random.randint(0, 255, (584, 565, 3), dtype=np.uint8)
    m = np.zeros((584, 565), dtype=np.float32)
    m[100:200, 100:200] = 1
    out = t(image=img, mask=m)
    print("image:", tuple(out["image"].shape), out["image"].dtype)
    print("mask :", tuple(out["mask"].shape), "unique:", np.unique(out["mask"]))
except Exception as e:
    print("Compose build FAIL:", type(e).__name__, e)

print()
print("=== CPU threads:", torch.get_num_threads(), "===")
torch.set_num_threads(8)

import segmentation_models_pytorch as smp

results = []
for size in (384, 512):
    for enc in ("resnet18", "resnet34"):
        model = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=3, classes=1)
        model.train()
        x = torch.randn(4, 3, size, size)
        y = torch.randn(4, 1, size, size)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()

        loss = lossf(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        n = 2
        t0 = time.time()
        for _ in range(n):
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / n

        steps_per_epoch = 4  # 16 images / batch 4
        epoch_s = dt * steps_per_epoch
        results.append((enc, size, dt, epoch_s))
        print(f"{enc:9s} {size:4d} | {dt:6.2f} s/step | epoch(16 imgs) {epoch_s:5.1f} s")

print()
print("=== projection (3 folds) ===")
for enc, size, dt, epoch_s in results:
    for ep in (40, 60, 80):
        total = epoch_s * ep * 3 / 60.0
        print(f"{enc:9s} {size:4d} {ep:3d} epochs x3 folds -> {total:5.1f} min")
