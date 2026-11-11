# -*- coding: utf-8 -*-
"""Generate the Kaggle notebook v2 (fundus vessel segmentation, B-plan)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "id": f"md-{len(cells):02d}",
        "metadata": {},
        "source": text.strip("\n"),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "id": f"code-{len(cells):02d}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n"),
    })


# ----------------------------------------------------------------------------
md(r"""
# 眼底血管分割 · Kaggle Notebook（v2 · B 方案）

> 上一版：CV 0.7770，提交 0.74。本版针对「训练量不足 + 推理端未调优」重做。

## 0. 本版相对 v1 的改动

| # | 改动 | 预期影响 |
|---|---|---|
| 1 | 完整可复现：`CUBLAS_WORKSPACE_CONFIG` + `cudnn.deterministic` + DataLoader worker 播种 + 环境指纹 | 合规，同 GPU 可复现 |
| 2 | 编码器 `resnet34` → **`efficientnet-b4`** | +0.02 ~ 0.04 |
| 3 | `batch 4→8`，`epochs 80→200`，加 warmup | 训练步数约 2 倍，收敛更充分 |
| 4 | 训练时保存 **OOF 概率图** | 为阈值/后处理提供无泄漏的调参依据 |
| 5 | 在 OOF 上扫**阈值 + 小连通域面积** | +0.01 ~ 0.02 |
| 6 | TTA 3 → **24**（4 旋转 × 2 翻转 × 3 尺度） | +0.01 ~ 0.02 |
| 7 | 推理加**小连通域过滤**，输出仍为 512×512 PNG | 降假阳性 |

## 1. 数据与提交规则（已实测，v1 结论不变）

| 项 | 实际情况 |
|---|---|
| 训练图 / 标签 | 各 20 张，565×584，标签灰度 |
| 标签像素值 | **血管=255、背景=0** |
| 血管占比 | 6.0% ~ 11.6% |
| 测试图 | 20 张，565×584 |

官方 `segmentation_to_csv.py` 的行为：转灰度 → resize 到 512×512 → **取所有 >0 的像素** → RLE（1-based、行优先）。

**三条铁律**（v1 已用真实标签验证过）：

1. 预测图必须 **血管=255、背景=0**。反过来会把背景编码进 RLE，Dice≈0。
2. 预测图必须 **512×512**。尺寸不符时 BILINEAR 插值会让像素数膨胀（实测 400×400 输入多出 12276 个假阳性）。
3. RLE 索引 **1-based、行优先**。

## 2. 关于可复现性

本 notebook 固定了：

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`（cuBLAS 决定性的前提）
- `torch.backends.cudnn.deterministic=True`、`benchmark=False`
- Python / NumPy / PyTorch / CUDA 全部种子，以及 DataLoader 每个 worker 的独立种子
- `torch.use_deterministic_algorithms(True, warn_only=True)`

**在同一个 GPU 型号 + 同一个 torch 版本下，结果可以复现。**
跨 GPU 型号（T4 ↔ P100）不保证 bit 级一致，因为浮点累加顺序由硬件/驱动决定——这是公开的物理限制，不是代码问题。

运行结束会打印一行 **环境指纹**，复现时对照这一行即可。

## 3. 预计耗时

- 训练：约 40 ~ 55 分钟（T4，5 折，200 epoch，早停 patience=40）
- 推理：约 3 分钟（24 TTA × 5 模型 × 20 图）
- 合计：**约 1 小时**

> 运行前请打开 **Internet**（要下载 `efficientnet-b4` 的 ImageNet 权重）。
""")

# ----------------------------------------------------------------------------
md(r"""
## 4. 环境准备
""")

code(r'''
!pip install -q segmentation-models-pytorch timm 2>/dev/null

import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
''')

# ----------------------------------------------------------------------------
md(r"""
## 5. 可复现性设置 + 环境指纹

**必须在 CUDA 初始化之前**设置 `CUBLAS_WORKSPACE_CONFIG`，否则 cuBLAS 会走非确定路径。
""")

code(r'''
import os

# 必须放在 import torch 之前（PyTorch 初始化 cuBLAS 时读取）
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import random
import math
import numpy as np
import torch

SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 关闭 cuDNN 自动算法选择：这是 v1 不确定性的主要来源
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

# 强制确定性算子；warn_only 避免个别算子直接中断训练
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    _det_mode = "warn_only"
except TypeError:
    torch.use_deterministic_algorithms(True)
    _det_mode = "strict"

print("=" * 60)
print("ENVIRONMENT FINGERPRINT (对照这一行判断是否同环境)")
print("=" * 60)
print(f"python      : {__import__('sys').version.split()[0]}")
print(f"torch       : {torch.__version__}")
print(f"cuda        : {torch.version.cuda}")
print(f"cudnn       : {torch.backends.cudnn.version()}")
print(f"gpu         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"cudnn.det   : {torch.backends.cudnn.deterministic}")
print(f"cudnn.bench : {torch.backends.cudnn.benchmark}")
print(f"det.algo    : {_det_mode}")
print(f"cublas cfg  : {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")
print(f"seed        : {SEED}")
''')

# ----------------------------------------------------------------------------
md(r"""
## 6. 导入与全局配置
""")

code(r'''
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 512          # 必须 512：官方脚本按 512x512 做 RLE
N_FOLDS = 5
EPOCHS = 200            # v1: 80
PATIENCE = 40           # v1: 25
BATCH_SIZE = 8          # v1: 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
NUM_WORKERS = 2

ENCODER = "efficientnet-b4"   # v1: resnet34
ENCODER_WEIGHTS = "imagenet"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
WORK.mkdir(parents=True, exist_ok=True)

print("device :", DEVICE)
print("encoder:", ENCODER)
print("work   :", WORK)
''')

# ----------------------------------------------------------------------------
md(r"""
## 7. 定位数据（自动搜索，不写死路径）
""")

code(r'''
def locate_data_root():
    search_bases = [Path("/kaggle/input"), Path("/content"), Path.cwd(), Path.cwd().parent]
    hits = []
    for base in search_bases:
        if not base.exists():
            continue
        for p in base.rglob("segmentation_to_csv.py"):
            hits.append(p.parent)
    if not hits:
        return None
    for h in hits:
        if (h / "train" / "label").is_dir():
            return h
    return hits[0]


DATA_ROOT = locate_data_root()
if DATA_ROOT is None:
    raise FileNotFoundError("未找到数据目录：请把含 segmentation_to_csv.py 的比赛数据挂载为 Kaggle Dataset。")

TRAIN_IMG_DIR = DATA_ROOT / "train" / "image"
TRAIN_LBL_DIR = DATA_ROOT / "train" / "label"
TEST_IMG_DIR = DATA_ROOT / "test" / "image"
OFFICIAL_SCRIPT = DATA_ROOT / "segmentation_to_csv.py"

for p in (TRAIN_IMG_DIR, TRAIN_LBL_DIR, TEST_IMG_DIR, OFFICIAL_SCRIPT):
    print(f"{str(p):70s} {p.exists()}")
''')

# ----------------------------------------------------------------------------
md(r"""
## 8. 文件列表与标签确认
""")

code(r'''
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_ids(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    return sorted(files, key=lambda x: int(os.path.splitext(x)[0]))


TRAIN_IDS = list_ids(TRAIN_IMG_DIR)
TEST_IDS = list_ids(TEST_IMG_DIR)

print("train:", len(TRAIN_IDS), TRAIN_IDS[:5])
print("test :", len(TEST_IDS), TEST_IDS[:5])

ratios = []
for name in TRAIN_IDS:
    lbl = cv2.imread(str(TRAIN_LBL_DIR / name), cv2.IMREAD_GRAYSCALE)
    ratios.append(float((lbl > 127).mean()))

print(f"vessel ratio: mean={np.mean(ratios):.4f} min={np.min(ratios):.4f} max={np.max(ratios):.4f}")
assert min(ratios) > 0.01, "标签疑似反色：血管应该占约 10%"
print("标签格式确认：血管=255、背景=0 ✔")
''')

# ----------------------------------------------------------------------------
md(r"""
## 9. 数据可视化
""")

code(r'''
fig, axes = plt.subplots(3, 3, figsize=(13, 12.5))
for i in range(3):
    name = TRAIN_IDS[i]
    rgb = cv2.cvtColor(cv2.imread(str(TRAIN_IMG_DIR / name)), cv2.COLOR_BGR2RGB)
    lbl = cv2.imread(str(TRAIN_LBL_DIR / name), cv2.IMREAD_GRAYSCALE)
    axes[i, 0].imshow(rgb); axes[i, 0].set_title(f"image {name} {rgb.shape[:2]}")
    axes[i, 1].imshow(lbl, cmap="gray"); axes[i, 1].set_title("label (vessel=255)")
    axes[i, 2].imshow(rgb)
    axes[i, 2].imshow(np.ma.masked_where(lbl < 128, lbl), cmap="autumn", alpha=0.6)
    axes[i, 2].set_title("overlay")
    for ax in axes[i]:
        ax.axis("off")
plt.tight_layout()
plt.show()
''')

# ----------------------------------------------------------------------------
md(r"""
## 10. 预处理与增强

绿通道 CLAHE（眼底血管对比度最高的通道），复制为 3 通道以复用 ImageNet 预训练权重。
""")

code(r'''
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_green(rgb_uint8):
    green = rgb_uint8[:, :, 1]
    green = _clahe.apply(green)
    return np.stack([green, green, green], axis=-1)


def get_train_transform(size=IMG_SIZE):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=35,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        ToTensorV2(),
    ])


def get_valid_transform(size=IMG_SIZE):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        ToTensorV2(),
    ])
''')

# ----------------------------------------------------------------------------
md(r"""
## 11. Dataset

注意：**始终返回文件名**，因为 OOF 概率图要按文件名对齐。
""")

code(r'''
class VesselDataset(Dataset):
    def __init__(self, ids, img_dir, lbl_dir=None, transform=None):
        self.ids = ids
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir) if lbl_dir is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        bgr = cv2.imread(str(self.img_dir / name))
        if bgr is None:
            raise FileNotFoundError(self.img_dir / name)
        rgb = enhance_green(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if self.lbl_dir is None:
            out = self.transform(image=rgb)
            return out["image"], name

        lbl = cv2.imread(str(self.lbl_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = (lbl > 127).astype(np.float32)
        out = self.transform(image=rgb, mask=mask)
        return out["image"], out["mask"].unsqueeze(0), name
''')

# ----------------------------------------------------------------------------
md(r"""
## 12. 模型、损失与指标
""")

code(r'''
def build_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,
    )
    return model.to(DEVICE)


dice_loss_fn = smp.losses.DiceLoss(mode="binary")
bce_loss_fn = nn.BCEWithLogitsLoss()


def dice_from_mask(pred_bin, target_bin, eps=1e-6):
    """pred_bin / target_bin: numpy 0/1，任意形状。"""
    pred_bin = pred_bin.astype(np.float32)
    target_bin = target_bin.astype(np.float32)
    inter = (pred_bin * target_bin).sum()
    return (2.0 * inter + eps) / (pred_bin.sum() + target_bin.sum() + eps)


def dice_from_logits(logits, target, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * target).sum()
    return ((2.0 * inter + eps) / (pred.sum() + target.sum() + eps)).item()
''')

# ----------------------------------------------------------------------------
md(r"""
## 13. 训练循环（含 OOF 保存）

每个 epoch 只更新 2 步（16 张 / batch 8），所以把 epoch 数拉到 200，
用 warmup 让前 5 轮学习率线性爬升，避免一开始就震荡。
""")

code(r'''
def make_loader(dataset, fold, shuffle):
    gen = torch.Generator()
    gen.manual_seed(SEED + fold)

    def worker_init_fn(worker_id):
        np.random.seed(SEED + fold * 1000 + worker_id)
        random.seed(SEED + fold * 1000 + worker_id)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        generator=gen,
        worker_init_fn=worker_init_fn,
    )


def lr_lambda(epoch_idx):
    if epoch_idx < WARMUP_EPOCHS:
        return float(epoch_idx + 1) / float(WARMUP_EPOCHS)
    progress = (epoch_idx - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_fold(fold, tr_ids, va_ids, oof_probs, oof_masks):
    tr_ds = VesselDataset(tr_ids, TRAIN_IMG_DIR, TRAIN_LBL_DIR, get_train_transform())
    va_ds = VesselDataset(va_ids, TRAIN_IMG_DIR, TRAIN_LBL_DIR, get_valid_transform())

    tr_ld = make_loader(tr_ds, fold, shuffle=True)
    va_ld = make_loader(va_ds, fold, shuffle=False)

    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dice = 0.0
    best_epoch = -1
    best_path = WORK / f"fold{fold}.pth"
    patience_cnt = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for imgs, masks, _ in tr_ld:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = bce_loss_fn(logits, masks) + dice_loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(tr_ds)

        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks, _ in va_ld:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                dices.append(dice_from_logits(logits, masks))
        val_dice = float(np.mean(dices))

        scheduler.step()   # LambdaLR 按 epoch 步进
        history.append({"epoch": epoch, "loss": tr_loss, "val_dice": val_dice,
                        "lr": optimizer.param_groups[0]["lr"]})

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            best_epoch = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"[fold {fold}] ep {epoch:03d} | loss {tr_loss:.4f} | "
                  f"val_dice {val_dice:.4f} | best {best_dice:.4f}")

        if patience_cnt >= PATIENCE:
            print(f"[fold {fold}] early stop at epoch {epoch}")
            break

    # ---- 用 best 权重对整个验证折重新推理，保存 OOF 概率 ----
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        for imgs, masks, names in va_ld:
            logits = model(imgs.to(DEVICE, non_blocking=True))
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            gts = masks.numpy()[:, 0]
            for i, n in enumerate(names):
                oof_probs[n] = probs[i]
                oof_masks[n] = gts[i]

    print(f"[fold {fold}] DONE best_dice={best_dice:.4f} (epoch {best_epoch}) -> {best_path}")
    return best_dice, best_epoch, history, best_path
''')

# ----------------------------------------------------------------------------
md(r"""
## 14. 五折训练
""")

code(r'''
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
splits = list(kf.split(TRAIN_IDS))

OOF_PROBS = {}
OOF_MASKS = {}

fold_results = []
all_histories = {}
t0 = time.time()

for fold, (tr_idx, va_idx) in enumerate(splits):
    tr_ids = [TRAIN_IDS[i] for i in tr_idx]
    va_ids = [TRAIN_IDS[i] for i in va_idx]
    print(f"\n{'=' * 70}\nFold {fold} | train {len(tr_ids)} | val {len(va_ids)}")
    best_dice, best_epoch, hist, path = train_one_fold(fold, tr_ids, va_ids, OOF_PROBS, OOF_MASKS)
    fold_results.append({"fold": fold, "best_dice": best_dice, "best_epoch": best_epoch})
    all_histories[fold] = hist

print(f"\ntotal training time: {(time.time() - t0) / 60:.1f} min")
df_folds = pd.DataFrame(fold_results)
print(df_folds)
print(f"\nCV mean Dice (thr=0.5, no post) = {df_folds['best_dice'].mean():.4f} "
      f"(+/- {df_folds['best_dice'].std():.4f})")
print(f"OOF samples: {len(OOF_PROBS)}")
''')

# ----------------------------------------------------------------------------
md(r"""
## 15. 训练曲线
""")

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for fold, hist in all_histories.items():
    ep = [h["epoch"] for h in hist]
    axes[0].plot(ep, [h["loss"] for h in hist], label=f"fold {fold}")
    axes[1].plot(ep, [h["val_dice"] for h in hist], label=f"fold {fold}")
    axes[2].plot(ep, [h["lr"] for h in hist], label=f"fold {fold}")
axes[0].set_title("train loss"); axes[0].set_xlabel("epoch")
axes[1].set_title("val dice"); axes[1].set_xlabel("epoch")
axes[2].set_title("learning rate"); axes[2].set_xlabel("epoch")
for a in axes:
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()
''')

# ----------------------------------------------------------------------------
md(r"""
## 16. 在 OOF 上扫描「阈值 + 小连通域面积」

这一步用**全部 20 个样本的 OOF 预测**来选参数，没有用到测试集，不存在泄漏。

血管是连续的长条结构，面积很小的连通域基本都是假阳性；但过滤太狠会切断细小分支，所以两个参数一起扫。
""")

code(r'''
def remove_small_components(mask_uint8, min_size):
    """去掉面积小于 min_size 的连通域（8 连通）。"""
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


def postprocess(prob, thr, min_size):
    m = (prob > thr).astype(np.uint8)
    return remove_small_components(m, min_size)


print(f"OOF samples: {len(OOF_PROBS)}")
assert len(OOF_PROBS) == len(TRAIN_IDS), "OOF 数量应等于训练集数量"

grid = []
for thr in np.round(np.arange(0.30, 0.76, 0.05), 2):
    for min_size in [0, 10, 30, 60, 120, 200]:
        ds = [dice_from_mask(postprocess(OOF_PROBS[n], thr, min_size), OOF_MASKS[n])
              for n in OOF_PROBS]
        grid.append({"thr": float(thr), "min_size": min_size, "dice": float(np.mean(ds))})

grid = pd.DataFrame(grid).sort_values("dice", ascending=False).reset_index(drop=True)
print(grid.head(12).to_string(index=False))

BEST_THR = float(grid.iloc[0]["thr"])
BEST_MIN_SIZE = int(grid.iloc[0]["min_size"])
BEST_OOF_DICE = float(grid.iloc[0]["dice"])

baseline = grid[(grid["thr"] == 0.5) & (grid["min_size"] == 0)]["dice"].iloc[0]
print(f"\nbaseline (thr=0.50, min_size=0): {baseline:.4f}")
print(f"BEST     (thr={BEST_THR:.2f}, min_size={BEST_MIN_SIZE}): {BEST_OOF_DICE:.4f}")
print(f"OOF 提升: {BEST_OOF_DICE - baseline:+.4f}")
''')

code(r'''
# 阈值敏感性曲线：看最优值是不是一个「尖峰」——尖峰说明是噪声，不应采信
sub_thr = grid[grid["min_size"] == BEST_MIN_SIZE].sort_values("thr")
plt.figure(figsize=(7, 4))
plt.plot(sub_thr["thr"], sub_thr["dice"], marker="o")
plt.axvline(BEST_THR, ls="--", color="r", label=f"best={BEST_THR:.2f}")
plt.xlabel("threshold"); plt.ylabel("OOF dice"); plt.grid(alpha=0.3); plt.legend()
plt.title(f"threshold sweep (min_size={BEST_MIN_SIZE})")
plt.show()
''')

# ----------------------------------------------------------------------------
md(r"""
## 17. 推理：5 折模型 + 24 路 TTA

TTA 组合：3 个尺度（0.75 / 1.0 / 1.25）× 4 个旋转（0/90/180/270）× 2 个翻转 = **24 路**。

多尺度的作用：细血管在不同尺度下响应不同，平均后能补上小尺度丢失的分支。
""")

code(r'''
def load_models():
    models = []
    for fold in range(N_FOLDS):
        path = WORK / f"fold{fold}.pth"
        if not path.exists():
            print("missing:", path)
            continue
        m = build_model()
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"loaded {len(models)} models")
    return models


@torch.no_grad()
def tta_predict(model, enhanced_rgb):
    """enhanced_rgb: (512,512,3) uint8。返回 (512,512) 概率图。"""
    base = enhanced_rgb.astype(np.float32) / 255.0
    acc = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE), device=DEVICE)
    n = 0

    for scale in (0.75, 1.0, 1.25):
        s = int(round(IMG_SIZE * scale / 32.0)) * 32
        if s < 64:
            continue
        im = cv2.resize(base, (s, s), interpolation=cv2.INTER_LINEAR)
        x0 = (im - MEAN) / STD
        x0 = torch.from_numpy(x0.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

        for k in range(4):
            x1 = torch.rot90(x0, k, dims=[2, 3])
            for flip in (False, True):
                x2 = torch.flip(x1, dims=[3]) if flip else x1
                p = torch.sigmoid(model(x2))
                if flip:
                    p = torch.flip(p, dims=[3])
                p = torch.rot90(p, -k, dims=[2, 3])
                if s != IMG_SIZE:
                    p = F.interpolate(p, size=(IMG_SIZE, IMG_SIZE),
                                      mode="bilinear", align_corners=False)
                acc += p
                n += 1

    return (acc / max(n, 1))[0, 0].cpu().numpy()


@torch.no_grad()
def predict_prob(models, bgr):
    rgb = enhance_green(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    probs = [tta_predict(m, rgb) for m in models]
    return np.mean(probs, axis=0)
''')

# ----------------------------------------------------------------------------
md(r"""
## 18. 生成 512×512 预测图

格式：**血管=255、背景=0**，PNG（无损，避免 JPEG 压缩产生灰度中间值）。
""")

code(r'''
MODELS = load_models()
assert len(MODELS) == N_FOLDS, f"模型数量不对: {len(MODELS)}"

PRED_DIR = WORK / "image"
if PRED_DIR.exists():
    for f in PRED_DIR.glob("*"):
        f.unlink()
PRED_DIR.mkdir(parents=True, exist_ok=True)

test_probs = {}
t0 = time.time()

for name in TEST_IDS:
    bgr = cv2.imread(str(TEST_IMG_DIR / name))
    prob = predict_prob(MODELS, bgr)
    test_probs[name] = prob

    mask = postprocess(prob, BEST_THR, BEST_MIN_SIZE).astype(np.uint8) * 255
    out_name = os.path.splitext(name)[0] + ".png"
    cv2.imwrite(str(PRED_DIR / out_name), mask)

print(f"saved {len(TEST_IDS)} predictions in {time.time() - t0:.1f}s")
print("thr:", BEST_THR, "| min_size:", BEST_MIN_SIZE)
print("preview:", sorted(os.listdir(PRED_DIR))[:5])
''')

code(r'''
_name = TEST_IDS[0]
_bgr = cv2.imread(str(TEST_IMG_DIR / _name))
_prob = test_probs[_name]
_final = postprocess(_prob, BEST_THR, BEST_MIN_SIZE)

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
axes[0].imshow(cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title("test image")
axes[1].imshow(_prob, cmap="gray"); axes[1].set_title("TTA probability")
axes[2].imshow(_final, cmap="gray"); axes[2].set_title(f"mask thr={BEST_THR} min={BEST_MIN_SIZE}")
axes[3].imshow(cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB))
axes[3].imshow(np.ma.masked_where(_final == 0, _final), cmap="autumn", alpha=0.7)
axes[3].set_title("overlay")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
''')

# ----------------------------------------------------------------------------
md(r"""
## 19. 调用官方脚本生成 `submission.csv`
""")

code(r'''
import shutil
import subprocess

shutil.copy(str(OFFICIAL_SCRIPT), str(WORK / "segmentation_to_csv.py"))

res = subprocess.run(
    [sys.executable, "segmentation_to_csv.py"],
    cwd=str(WORK),
    capture_output=True,
    text=True,
)
print("returncode:", res.returncode)
print("stdout:", res.stdout[-2000:])
print("stderr:", res.stderr[-2000:])

SUB_PATH = WORK / "submission.csv"
print("submission exists:", SUB_PATH.exists(), SUB_PATH)
''')

# ----------------------------------------------------------------------------
md(r"""
## 20. 提交校验

除了 v1 的合法性检查，这一版额外做两件事：

1. **按 Id 排序重写**，与 `sample_submission.csv` 的行顺序完全对齐（零风险）
2. **统计每张图的血管占比**，偏离 0.03~0.20 就报警——这是发现「编码方向反了」最快的信号
""")

code(r'''
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


def rle_decode(rle_str, h=IMG_SIZE, w=IMG_SIZE):
    mask = np.zeros(h * w, dtype=np.uint8)
    if isinstance(rle_str, str) and rle_str.strip():
        vals = list(map(int, rle_str.split()))
        assert len(vals) % 2 == 0, "像素对数量必须为偶数"
        for i in range(0, len(vals), 2):
            start, length = vals[i], vals[i + 1]
            assert start >= 1, f"起始位置必须为正数: {start}"
            assert length >= 1, f"长度必须为正数: {length}"
            assert start - 1 + length <= h * w, "越界"
            mask[start - 1:start - 1 + length] = 1
    return mask.reshape(h, w)


df_sub = pd.read_csv(SUB_PATH)
df_sub["Id"] = df_sub["Id"].astype(int)
df_sub = df_sub.sort_values("Id").reset_index(drop=True)

assert len(df_sub) == len(TEST_IDS), f"行数不符: {len(df_sub)} vs {len(TEST_IDS)}"
assert set(df_sub["Id"]) == set(int(os.path.splitext(x)[0]) for x in TEST_IDS), "Id 不匹配"

print(f"{'Id':>4} {'pairs':>8} {'ratio':>8}  status")
print("-" * 40)
ratios_out = []
for _, row in df_sub.iterrows():
    rid = int(row["Id"])
    rle = "" if pd.isna(row["Predicted"]) else str(row["Predicted"])
    decoded = rle_decode(rle)
    ratio = decoded.sum() / (IMG_SIZE * IMG_SIZE)
    ratios_out.append(ratio)

    src = None
    for t in TEST_IDS:
        if int(os.path.splitext(t)[0]) == rid:
            src = t
            break
    our = postprocess(test_probs[src], BEST_THR, BEST_MIN_SIZE)
    status = "ok" if np.array_equal(decoded, our) else "MISMATCH"
    print(f"{rid:>4} {len(rle.split()) // 2:>8} {ratio:>8.4f}  {status}")

print("-" * 40)
print(f"ratio: mean={np.mean(ratios_out):.4f} min={np.min(ratios_out):.4f} max={np.max(ratios_out):.4f}")
print(f"train 集血管占比参考: mean={np.mean(ratios):.4f}")

if min(ratios_out) < 0.02 or max(ratios_out) > 0.30:
    print("⚠️ 警告：有图的血管占比异常，检查编码方向（应为血管=255）")
else:
    print("✔ 血管占比在合理范围内，编码方向正确")

# 按 Id 排序后覆盖
df_sub.to_csv(SUB_PATH, index=False)
print("\n已按 Id 排序写回:", SUB_PATH)
print(df_sub[["Id"]].T)
''')

# ----------------------------------------------------------------------------
md(r"""
## 21. 提交与环境说明

**提交**：下载 `/kaggle/working/submission.csv`，点 `Submit to Competition`。

### 复现说明（比赛要求）

本 notebook 已经固定：

| 设置 | 值 |
|---|---|
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| `cudnn.deterministic` | `True` |
| `cudnn.benchmark` | `False` |
| `torch.use_deterministic_algorithms` | `True (warn_only)` |
| 全局种子 | 42（Python / NumPy / Torch / CUDA / 每个 DataLoader worker） |

**同 GPU 型号 + 同 torch 版本下可复现**；跨 GPU 型号不保证 bit 级一致。
运行结束时把「ENVIRONMENT FINGERPRINT」那一段一并保存，作为复现依据。

### 如果这一版还没到 0.8

按性价比继续：

| 优先级 | 措施 | 预期 |
|---|---|---|
| 1 | 看第 16 节扫出来的 `BEST_THR` 是否明显偏离 0.5；偏离大说明 v1 主要亏在阈值上 | — |
| 2 | 把 `IMG_SIZE` 提到 640，推理后用 `INTER_NEAREST` 缩回 512 再存（**不能用 BILINEAR**） | +0.01~0.02 |
| 3 | 再训一个 `mit_b3` 模型，两套概率平均 | +0.02~0.03 |
| 4 | 伪标签：用本版模型预测 test，置信度 >0.9 / <0.1 的像素当标签，和 train 一起重训 | +0.02~0.05 |
| 5 | 10 折（每折 2 张验证，方差更大，收益不稳定） | 不推荐 |

### 不要做的事

- ❌ 输出「血管=0、背景=255」——官方脚本会把背景编码进 RLE
- ❌ 预测图尺寸不是 512×512——resize 插值会造成边界膨胀
- ❌ 用 JPEG 保存预测图——有损压缩产生的灰度中间值会被 `>0` 误判
- ❌ 用测试集调阈值——只能看 OOF，否则就是变相刷榜
""")

# ----------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(r"E:\machine learing\project4\vessel_segmentation_kaggle.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, out.stat().st_size, "bytes,", len(cells), "cells")
