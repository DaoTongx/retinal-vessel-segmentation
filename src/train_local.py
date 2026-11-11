# -*- coding: utf-8 -*-
"""
DRIVE fundus vessel segmentation - LOCAL CPU version.

Usage:
    python train_local.py                     # full run: 5 folds, 80 epochs, 512px
    python train_local.py --folds 3 --epochs 60   # faster
    python train_local.py --predict-only      # use existing fold*.pth, skip training
    python train_local.py --resume            # skip folds that already have weights

Outputs (under --out, default ./run):
    fold{0..4}.pth          model weights
    oof.npz                 OOF probability maps + masks
    logs/*.log              training log
    predictions/*.png       512x512 masks (vessel=255)
    submission.csv          final submission (sorted by Id)
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
# 国内网络下 huggingface.co 不可达，默认走镜像（smp 通过 HF Hub 取 encoder 权重）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SEED = 42
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg, fp=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def locate_data_root():
    here = Path(__file__).resolve().parent
    candidates = [here / "data" / "segmentation", here.parent / "data" / "segmentation"]
    for c in candidates:
        if (c / "train" / "label").is_dir():
            return c
    for p in here.rglob("segmentation_to_csv.py"):
        if (p.parent / "train" / "label").is_dir():
            return p.parent
    return None


def list_ids(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    return sorted(files, key=lambda x: int(os.path.splitext(x)[0]))


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_green(rgb_uint8):
    green = _clahe.apply(rgb_uint8[:, :, 1])
    return np.stack([green, green, green], axis=-1)


def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.0625, 0.0625),
                 rotate=(-35, 35), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        ToTensorV2(),
    ])


def get_valid_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        ToTensorV2(),
    ])


class VesselDataset(Dataset):
    def __init__(self, ids, img_dir, lbl_dir=None, transform=None):
        self.ids = ids
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir) if lbl_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        bgr = cv2.imread(str(self.img_dir / name))
        rgb = enhance_green(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if self.lbl_dir is None:
            out = self.transform(image=rgb)
            return out["image"], name

        lbl = cv2.imread(str(self.lbl_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = (lbl > 127).astype(np.float32)
        out = self.transform(image=rgb, mask=mask)
        return out["image"], out["mask"].unsqueeze(0), name


# --------------------------------------------------------------------------- #
# FOV mask + postprocess
# --------------------------------------------------------------------------- #
def make_fov_mask(rgb, size, thresh=10):
    """Binary circle-ish retina area from brightness, holes filled."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = ((lab == idx) * 255).astype(np.uint8)

    h, w = th.shape
    ff = th.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    th = th | cv2.bitwise_not(ff)
    return th


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


def postprocess(prob, thr, min_size, fov=None):
    m = (prob > thr).astype(np.uint8)
    if fov is not None:
        m = m * (fov > 0).astype(np.uint8)
    return remove_small_components(m, min_size)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def dice_np(pred, target, eps=1e-6):
    p = pred.astype(np.float32)
    t = target.astype(np.float32)
    inter = (p * t).sum()
    return float((2.0 * inter + eps) / (p.sum() + t.sum() + eps))


def build_model(encoder, device):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet",
                    in_channels=3, classes=1).to(device)


def train_one_fold(args, fold, tr_ids, va_ids, dirs, device, fp):
    size = args.size
    tr_ds = VesselDataset(tr_ids, dirs["train_img"], dirs["train_lbl"], get_train_transform(size))
    va_ds = VesselDataset(va_ids, dirs["train_img"], dirs["train_lbl"], get_valid_transform(size))

    gen = torch.Generator()
    gen.manual_seed(SEED + fold)

    def worker_init(worker_id):
        np.random.seed(SEED + fold * 1000 + worker_id)
        random.seed(SEED + fold * 1000 + worker_id)

    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                       pin_memory=False, drop_last=False, generator=gen,
                       worker_init_fn=worker_init)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0,
                       pin_memory=False)

    model = build_model(args.encoder, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(e):
        if e < args.warmup:
            return float(e + 1) / float(args.warmup)
        p = (e - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    bce = nn.BCEWithLogitsLoss()
    dloss = smp.losses.DiceLoss(mode="binary")

    best = 0.0
    best_ep = -1
    best_path = dirs["out"] / f"fold{fold}.pth"
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for imgs, masks, _ in tr_ld:
            imgs = imgs.to(device)
            masks = masks.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = bce(logits, masks) + dloss(logits, masks)
            loss.backward()
            opt.step()
            tot += loss.item() * imgs.size(0)
        tot /= len(tr_ds)
        sched.step()

        model.eval()
        ds = []
        with torch.no_grad():
            for imgs, masks, _ in va_ld:
                imgs = imgs.to(device)
                masks = masks.to(device)
                pred = (torch.sigmoid(model(imgs)) > 0.5).float()
                for i in range(imgs.size(0)):
                    ds.append(dice_np(pred[i, 0].cpu().numpy(), masks[i, 0].cpu().numpy()))
        val = float(np.mean(ds))

        if val > best:
            best, best_ep = val, ep
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1

        if ep == 1 or ep % 5 == 0 or val > best - 1e-9:
            log(f"[fold {fold}] ep {ep:03d} | loss {tot:.4f} | dice {val:.4f} | best {best:.4f}", fp)

        if patience >= args.patience:
            log(f"[fold {fold}] early stop at ep {ep}", fp)
            break

    log(f"[fold {fold}] DONE best={best:.4f} @ep{best_ep} -> {best_path.name}", fp)
    return best, best_path


@torch.no_grad()
def oof_predict(model, ds, device, batch=4):
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    model.eval()
    probs, gts = {}, {}
    for imgs, masks, names in ld:
        p = torch.sigmoid(model(imgs.to(device))).cpu().numpy()[:, 0]
        g = masks.numpy()[:, 0]
        for i, n in enumerate(names):
            probs[n] = p[i]
            gts[n] = g[i]
    return probs, gts


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def tta_predict(model, enhanced, size, device):
    base = enhanced.astype(np.float32) / 255.0
    x0 = (base - MEAN) / STD
    x0 = torch.from_numpy(x0.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    acc = torch.zeros((1, 1, size, size), device=device)
    n = 0
    for k in range(4):
        x1 = torch.rot90(x0, k, dims=[2, 3])
        for flip in (False, True):
            x2 = torch.flip(x1, dims=[3]) if flip else x1
            p = torch.sigmoid(model(x2))
            if flip:
                p = torch.flip(p, dims=[3])
            p = torch.rot90(p, -k, dims=[2, 3])
            acc += p
            n += 1
    return (acc / n)[0, 0].cpu().numpy()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--out", default="run")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    seed_everything()
    torch.set_num_threads(args.threads)

    data_root = locate_data_root()
    if data_root is None:
        raise SystemExit("data root not found (need train/label next to segmentation_to_csv.py)")

    dirs = {
        "root": data_root,
        "train_img": data_root / "train" / "image",
        "train_lbl": data_root / "train" / "label",
        "test_img": data_root / "test" / "image",
        "script": data_root / "segmentation_to_csv.py",
        "out": Path(args.out),
    }
    for k in ("train_img", "train_lbl", "test_img"):
        if not dirs[k].is_dir():
            raise SystemExit(f"missing dir: {dirs[k]}")

    (dirs["out"] / "logs").mkdir(parents=True, exist_ok=True)
    (dirs["out"] / "predictions").mkdir(parents=True, exist_ok=True)

    fp = open(dirs["out"] / "logs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log", "w", encoding="utf-8")

    device = torch.device("cpu")
    train_ids = list_ids(dirs["train_img"])
    test_ids = list_ids(dirs["test_img"])

    log("=" * 70, fp)
    log(f"data_root : {data_root}", fp)
    log(f"device    : cpu | threads {args.threads} | torch {torch.__version__}", fp)
    log(f"encoder   : {args.encoder} | size {args.size} | batch {args.batch}", fp)
    log(f"folds {args.folds} | epochs {args.epochs} | train {len(train_ids)} | test {len(test_ids)}", fp)
    log("=" * 70, fp)

    t_start = time.time()
    bests = {}
    oof_probs, oof_gts = {}, {}

    if not args.predict_only:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        splits = list(kf.split(train_ids))

        for fold, (tri, vai) in enumerate(splits):
            tr_ids = [train_ids[i] for i in tri]
            va_ids = [train_ids[i] for i in vai]
            ckpt = dirs["out"] / f"fold{fold}.pth"

            if args.resume and ckpt.exists():
                log(f"[fold {fold}] resume: found {ckpt.name}, skip training", fp)
                model = build_model(args.encoder, device)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                va_ds = VesselDataset(va_ids, dirs["train_img"], dirs["train_lbl"],
                                      get_valid_transform(args.size))
                p, g = oof_predict(model, va_ds, device, args.batch)
                oof_probs.update(p)
                oof_gts.update(g)
                continue

            log("", fp)
            log(f"Fold {fold} | train {len(tr_ids)} | val {len(va_ids)}", fp)
            best, ckpt = train_one_fold(args, fold, tr_ids, va_ids, dirs, device, fp)
            bests[fold] = best

            model = build_model(args.encoder, device)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            va_ds = VesselDataset(va_ids, dirs["train_img"], dirs["train_lbl"],
                                  get_valid_transform(args.size))
            p, g = oof_predict(model, va_ds, device, args.batch)
            oof_probs.update(p)
            oof_gts.update(g)

        if bests:
            log("", fp)
            log(f"CV mean dice = {np.mean(list(bests.values())):.4f} "
                f"(+/- {np.std(list(bests.values())):.4f})", fp)

    np.savez_compressed(dirs["out"] / "oof.npz",
                        probs=np.array([oof_probs[k] for k in sorted(oof_probs)]),
                        gts=np.array([oof_gts[k] for k in sorted(oof_gts)]),
                        names=np.array(sorted(oof_probs)))

    # ---- threshold / min_size sweep on OOF ----
    best_cfg = None
    rows = []
    for thr in np.round(np.arange(0.30, 0.76, 0.05), 2):
        for ms in (0, 30, 60, 120):
            ds = []
            for k in oof_probs:
                m = (oof_probs[k] > thr).astype(np.uint8)
                m = remove_small_components(m, ms)
                ds.append(dice_np(m, oof_gts[k]))
            rows.append((float(thr), ms, float(np.mean(ds))))
    rows.sort(key=lambda r: -r[2])
    best_cfg = rows[0]
    base = [r for r in rows if r[0] == 0.5 and r[1] == 0][0]
    log("", fp)
    log(f"OOF sweep best: thr={best_cfg[0]:.2f} min_size={best_cfg[1]} dice={best_cfg[2]:.4f}", fp)
    log(f"OOF baseline(0.50,0): dice={base[2]:.4f} | gain {best_cfg[2]-base[2]:+.4f}", fp)
    log("top5: " + " | ".join(f"({r[0]:.2f},{r[1]},{r[2]:.4f})" for r in rows[:5]), fp)

    THR, MIN_SIZE = best_cfg[0], best_cfg[1]

    # ---- inference on test ----
    models = []
    for fold in range(args.folds):
        ckpt = dirs["out"] / f"fold{fold}.pth"
        if not ckpt.exists():
            log(f"missing {ckpt.name}, skip", fp)
            continue
        m = build_model(args.encoder, device)
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        m.eval()
        models.append(m)
    log("", fp)
    log(f"loaded {len(models)} models for test inference", fp)

    pred_dir = dirs["out"] / "predictions"
    for f in pred_dir.glob("*"):
        f.unlink()

    t0 = time.time()
    for name in test_ids:
        bgr = cv2.imread(str(dirs["test_img"] / name))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        enh = enhance_green(rgb)
        enh = cv2.resize(enh, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        fov = make_fov_mask(rgb, args.size)

        probs = [tta_predict(m, enh, args.size, device) for m in models]
        prob = np.mean(probs, axis=0)

        mask = postprocess(prob, THR, MIN_SIZE, fov).astype(np.uint8) * 255
        cv2.imwrite(str(pred_dir / (os.path.splitext(name)[0] + ".png")), mask)
        ratio = (mask > 0).mean()
        log(f"  {name:10s} vessel_ratio={ratio:.4f}", fp)

    log(f"test inference done in {time.time()-t0:.1f}s", fp)

    # ---- official csv script ----
    import shutil
    shutil.copy(str(dirs["script"]), str(dirs["out"] / "segmentation_to_csv.py"))
    work = dirs["out"]
    # official script expects ./image
    img_link = work / "image"
    if img_link.exists():
        shutil.rmtree(img_link)
    shutil.copytree(pred_dir, img_link)

    res = subprocess.run([sys.executable, "segmentation_to_csv.py"], cwd=str(work),
                         capture_output=True, text=True)
    log(f"official script rc={res.returncode}", fp)
    if res.returncode != 0:
        log(res.stdout[-2000:], fp)
        log(res.stderr[-2000:], fp)

    sub_path = work / "submission.csv"
    df = pd.read_csv(sub_path)
    df["Id"] = df["Id"].astype(int)
    df = df.sort_values("Id").reset_index(drop=True)
    df.to_csv(sub_path, index=False)

    log("", fp)
    log(f"submission -> {sub_path}  ({len(df)} rows)", fp)
    log(f"total elapsed {(time.time()-t_start)/60:.1f} min", fp)
    fp.close()


if __name__ == "__main__":
    main()
