# -*- coding: utf-8 -*-
"""Generate the VascX probe notebook."""
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
# VascX 探针（只验证流程，不训练）

## 目的

DRIVE 上要做到 0.83，靠我们自己 16 张图训练已经到头了（你当前 0.76）。
最直接的杠杆是换用**在多个眼底数据集上训练过的现成模型**，我选的是
[VascX](https://pypi.org/project/retinalysis-vascx/)（[GitHub](https://github.com/Eyened/retinalysis-vascx)，
权重在 [HuggingFace](https://huggingface.co/Eyened/vascx)）。

但它的流程和我们之前完全不同：**先把视野裁成 1024×1024 的方形再推理**，
所以输出必须用 `bounds.csv` 反向映射回原图坐标。这一步搞错，结果就全废。

这个 notebook 只做一件事：**用 2 张图把整条链路走通**，确认三件事：

| # | 验收点 | 失败时怎么办 |
|---|---|---|
| 1 | `retinalysis-vascx` 能装成功，且 CUDA 没被破坏 | 换 `--no-deps` 或放弃路线 B |
| 2 | 权重能下载、能推理出分割图 | 手动 `--model-dir` 指定权重 |
| 3 | `bounds.csv` 能还原回原图坐标 | 手工推导映射公式 |

**三项全过 → 再谈全量集成。任一项失败 → 不浪费时间，回退到 FOV 后处理方案。**

> 前提：Kaggle 设置里 **Internet 必须打开**。
""")

# ----------------------------------------------------------------------------
md(r"""
## 1. 装之前先记下环境

`pip install` 有可能把 Kaggle 自带的 CUDA 版 torch 降级成 CPU 版，那样 GPU 就废了。
所以装之前先存一份指纹。
""")

code(r'''
import sys
import subprocess
import time

try:
    import torch
    print("BEFORE  torch :", torch.__version__)
    print("BEFORE  cuda  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("BEFORE  gpu   :", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", e)

print("python:", sys.version.split()[0])
''')

# ----------------------------------------------------------------------------
md(r"""
## 2. 安装 retinalysis-vascx

依赖较重（MONAI、retinalysis-enface、retinalysis-inference 等），预计 3~8 分钟。
装完不急着下结论，下一步会检查 torch 是否还在。
""")

code(r'''
t0 = time.time()
proc = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "retinalysis-vascx"],
    capture_output=True,
    text=True,
    timeout=2400,
)
elapsed = time.time() - t0
print(f"elapsed: {elapsed:.1f}s")
print("returncode:", proc.returncode)
print("--- stdout (tail) ---")
print(proc.stdout[-1500:])
print("--- stderr (tail) ---")
print(proc.stderr[-4000:])
''')

# ----------------------------------------------------------------------------
md(r"""
## 2.5 补装核心推理库（关键一步）

**这一步是因为环境墙**：`retinalysis-inference` 在 PyPI 上声明 `Requires: Python <3.13`，
而 Kaggle 现在是 Python 3.13，pip 会**静默跳过**这个依赖，结果 `vascx` 只装了个空壳，
一跑就报 `No module named 'rtnls_inference'`。

这里用 `--ignore-requires-python` 绕过版本检查强装。它是纯 Python 包，
3.13 上大概率没问题；如果 import 仍然失败，再看第 3 节的提示走 conda 方案。
""")

code(r'''
import sys
import importlib.metadata as md

print("python:", sys.version.split()[0])
for pkg in ["retinalysis-vascx", "retinalysis-inference", "monai"]:
    try:
        print(f"{pkg:26s} installed: {md.version(pkg)}")
    except Exception:
        print(f"{pkg:26s} not installed")

print()
print("vascx 声明的依赖:")
for r in (md.metadata("retinalysis-vascx").get_all("Requires-Dist") or []):
    if "inference" in r.lower() or "monai" in r.lower() or "enface" in r.lower():
        print("   ", r)

t0 = time.time()
proc = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "--ignore-requires-python",
     "retinalysis-vascx",
     "retinalysis-inference==0.8.0",   # vascx 2.1.x 依赖这个版本，1.0.0 移除了 dtos_inference
     "monai"],
    capture_output=True, text=True, timeout=2400,
)
print()
print(f"elapsed: {time.time() - t0:.1f}s")
print("returncode:", proc.returncode)
print("--- stdout (tail) ---")
print(proc.stdout[-1500:])
print("--- stderr (tail) ---")
print(proc.stderr[-4000:])
''')

code(r'''
# 关键验证：vascx 实际 import 的就是这一行
import importlib

for mod in ["rtnls_inference", "monai", "vascx"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:20s} OK   path={getattr(m, '__file__', '?')}")
    except Exception as e:
        print(f"{mod:20s} FAIL {type(e).__name__}: {e}")

print()
try:
    from rtnls_inference.dtos_inference import ModelInputDTO
    print("dtos_inference.ModelInputDTO  OK ->", ModelInputDTO)
except Exception as e:
    print("dtos_inference FAIL:", type(e).__name__, e)

# 顺便看看这个包里到底有哪些子模块，便于判断版本是否正确
try:
    import pkgutil
    import rtnls_inference
    subs = sorted(m.name for m in pkgutil.iter_modules(rtnls_inference.__path__))
    print("rtnls_inference submodules:", subs)
except Exception as e:
    print("list submodules failed:", e)

print()
print("rtnls_inference version:", getattr(__import__("rtnls_inference"), "__version__", "?"))
''')

# ----------------------------------------------------------------------------
md(r"""
## 3. 安装后体检

这一步是**整个探针最关键的分叉点**。

- 如果 `torch.cuda.is_available()` 变成 `False` → 依赖把 torch 换成了 CPU 版，必须立刻修
- 如果 `rtnls_inference` 仍然 FAIL → 说明 3.13 上真的跑不了，改走 conda 方案（见本节末尾）
""")

code(r'''
import importlib
import shutil

for mod in ["vascx", "rtnls_inference", "monai"]:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"{mod:20s} OK   version={ver}")
    except Exception as e:
        print(f"{mod:20s} FAIL {type(e).__name__}: {e}")

print()
print("vascx CLI path:", shutil.which("vascx"))

# 用干净子进程检查 torch：主进程里的 torch 可能是装包前的旧实例，
# 直接 reload 反而可能重新初始化 CUDA 导致卡死。
probe = subprocess.run(
    [sys.executable, "-c",
     "import torch; print('VERSION', torch.__version__); "
     "print('CUDA', torch.cuda.is_available()); "
     "print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"],
    capture_output=True, text=True,
)
print(probe.stdout)
if probe.returncode != 0:
    print("--- stderr ---")
    print(probe.stderr[-2000:])
''')

# ----------------------------------------------------------------------------
md(r"""
### 如果 CUDA 被破坏

在 Kaggle 上通常是因为 `retinalysis-*` 锁定了某个 `torch==2.x`（CPU 索引）。
修复方式是按 Kaggle 原始 CUDA 版本重装 torch。执行前先看报错里实际装进来的版本号：

```python
# 示例：只在确认被降级后执行
# !pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

不急着改，先看下一步 `vascx run-models` 能不能跑起来。
""")

# ----------------------------------------------------------------------------
md(r"""
### 如果方案 A 仍然 FAIL：conda 方案（约 15 分钟）

只有当上面 `rtnls_inference` 仍然 import 失败时才走这条路。

```bash
%%bash
conda create -y -n vascx python=3.11
source /opt/conda/etc/profile.d/conda.sh
conda activate vascx
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q retinalysis-vascx
```

然后用 conda 环境里的解释器跑（注意路径）：

```python
CONDA_BIN = "/opt/conda/envs/vascx/bin"
cmd = [f"{CONDA_BIN}/vascx", "run-models", str(IMG_IN), str(OUT),
       "--device", "cuda:0", "--no-disc", "--no-fovea", "--no-quality", "--no-overlay"]
subprocess.run(cmd, capture_output=True, text=True)
```

代价：要重新下载约 2.5GB 的 torch，而且 conda 环境里需要自己确认 CUDA 可见。
如果这一步也走不通，我会直接放弃 VascX，转 FOV + 多模型集成方案。
""")

# ----------------------------------------------------------------------------
md(r"""
## 4. 定位数据并挑 2 张测试图
""")

code(r'''
from pathlib import Path
import os
import cv2
import numpy as np


def locate_data_root():
    for base in [Path("/kaggle/input"), Path("/content"), Path.cwd(), Path.cwd().parent]:
        if not base.exists():
            continue
        for p in base.rglob("segmentation_to_csv.py"):
            if (p.parent / "train" / "label").is_dir():
                return p.parent
    return None


DATA_ROOT = locate_data_root()
print("DATA_ROOT:", DATA_ROOT)
assert DATA_ROOT is not None, "没找到数据目录"

TEST_IMG_DIR = DATA_ROOT / "test" / "image"
TEST_IDS = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")],
                  key=lambda x: int(os.path.splitext(x)[0]))
print("test:", len(TEST_IDS), TEST_IDS[:5])
''')

code(r'''
import shutil

WORK = Path("/kaggle/working")
PROBE = WORK / "vascx_probe"
IMG_IN = PROBE / "images"
OUT = PROBE / "segmentations"

for d in (IMG_IN, OUT):
    d.mkdir(parents=True, exist_ok=True)

TWO = TEST_IDS[:2]
for name in TWO:
    src = TEST_IMG_DIR / name
    dst = IMG_IN / name
    if not dst.exists():
        shutil.copy(str(src), str(dst))

print("probe images:", sorted(os.listdir(IMG_IN)))
for n in TWO:
    im = cv2.imread(str(IMG_IN / n))
    print(f"  {n}: shape={im.shape}")
''')

# ----------------------------------------------------------------------------
md(r"""
## 5. 跑 VascX 推理（只跑血管分支）

用 `--no-disc --no-fovea --no-quality --no-overlay` 跳过无关模型，
只留 vessels（视盘/黄斑/质量/叠加图对本题没用）。首次运行会从 HuggingFace 下载权重。
""")

code(r'''
t0 = time.time()

# CLI 入口可能叫 vascx（可执行文件），也可能只有 python -m 形式
vascx_exe = shutil.which("vascx")
if vascx_exe:
    cmd = [vascx_exe, "run-models", str(IMG_IN), str(OUT)]
    print("using CLI:", vascx_exe)
else:
    cmd = [sys.executable, "-m", "vascx", "run-models", str(IMG_IN), str(OUT)]
    print("CLI not found, trying python -m vascx")

cmd += ["--device", "cuda:0",
        "--no-disc", "--no-fovea", "--no-quality", "--no-overlay"]

print("cmd:", " ".join(cmd))

try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    print(f"elapsed: {time.time() - t0:.1f}s")
    print("returncode:", proc.returncode)
    print("--- stdout ---")
    print(proc.stdout[-3000:])
    print("--- stderr ---")
    print(proc.stderr[-4000:])
except subprocess.TimeoutExpired:
    print("TIMEOUT after 1h")
except Exception as e:
    print("EXEC FAILED:", type(e).__name__, e)
''')

# ----------------------------------------------------------------------------
md(r"""
## 6. 检查输出结构

VascX 的约定输出（参照官方 README）：

```
segmentations/
  - preprocessed_rgb/   # 1024x1024 的方形视野裁剪
  - vessels/            # 血管分割
  - artery_vein/
  - bounds.csv          # 视野裁剪边界
```
""")

code(r'''
print("OUT exists:", OUT.exists())
if OUT.exists():
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  {p.stat().st_size} bytes")

print()
for sub in ["preprocessed_rgb", "vessels", "artery_vein"]:
    d = OUT / sub
    if d.exists():
        print(f"{sub}: {sorted(os.listdir(d))}")
    else:
        print(f"{sub}: (不存在)")
''')

# ----------------------------------------------------------------------------
md(r"""
## 7. 读 `bounds.csv`——这是坐标映射的全部依据
""")

code(r'''
import pandas as pd

bounds_path = OUT / "bounds.csv"
if bounds_path.exists():
    df_bounds = pd.read_csv(bounds_path)
    print("columns:", list(df_bounds.columns))
    print("shape:", df_bounds.shape)
    print(df_bounds.to_string())
else:
    print("bounds.csv 不存在！")
    df_bounds = None
''')

# ----------------------------------------------------------------------------
md(r"""
## 8. 检查 vessels 输出图

要弄清楚三件事：**尺寸**、**值域（0/255 还是 0/1）**、**和原图的对应关系**。
""")

code(r'''
vessel_dir = OUT / "vessels"
if vessel_dir.exists():
    for f in sorted(os.listdir(vessel_dir)):
        arr = cv2.imread(str(vessel_dir / f), cv2.IMREAD_UNCHANGED)
        if arr is None:
            print(f"{f}: 读取失败")
            continue
        uniq = np.unique(arr)
        print(f"{f}: shape={arr.shape} dtype={arr.dtype} "
              f"min={arr.min()} max={arr.max()} unique={uniq[:8]} "
              f"positive_ratio={(arr > 0).mean():.4f}")
else:
    print("vessels 目录不存在")
''')

# ----------------------------------------------------------------------------
md(r"""
## 9. 映射回原图并可视化

这一步推断 `bounds.csv` 的含义。VascX 的预处理通常是：
**从原图裁出包含视野的方形区域 → resize 到 1024**。

如果 `bounds.csv` 给出裁剪框，映射就是：把 1024 的分割 resize 到裁剪框尺寸，再贴回原图。
""")

code(r'''
def guess_bounds_columns(df):
    """尽量兼容不同的列名写法。"""
    cols = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "x": ["x", "x0", "left", "x_min", "min_x", "h_start"],
        "y": ["y", "y0", "top", "y_min", "min_y", "w_start"],
        "w": ["w", "width", "w_size", "x_max"],
        "h": ["h", "height", "h_size", "y_max"],
    }
    found = {}
    for key, options in aliases.items():
        for o in options:
            if o in cols:
                found[key] = cols[o]
                break
    return found


def map_vessel_to_original(vessel_img, orig_h, orig_w, bounds=None):
    """把 VascX 的分割图映射回原图尺寸，返回 0/1 uint8。"""
    v = (vessel_img > 0).astype(np.uint8)

    if bounds is None:
        return cv2.resize(v, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    x, y, w, h = bounds
    small = cv2.resize(v, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((orig_h, orig_w), dtype=np.uint8)
    x, y = int(max(0, x)), int(max(0, y))
    w = int(min(w, orig_w - x))
    h = int(min(h, orig_h - y))
    if w > 0 and h > 0:
        out[y:y + h, x:x + w] = small[:h, :w]
    return out


if df_bounds is not None:
    print("bounds columns:", list(df_bounds.columns))
    print(df_bounds.head().to_string())
    guess = guess_bounds_columns(df_bounds)
    print("guessed key columns:", guess)
''')

code(r'''
import matplotlib.pyplot as plt

vessel_files = sorted(os.listdir(vessel_dir))[:2] if vessel_dir.exists() else []
id_col_candidates = [c for c in (df_bounds.columns if df_bounds is not None else []) if "id" in c.lower()]
id_col = id_col_candidates[0] if id_col_candidates else None
print("id column:", id_col)

guess = guess_bounds_columns(df_bounds) if df_bounds is not None else {}

for vf in vessel_files:
    stem = os.path.splitext(vf)[0]
    varr = cv2.imread(str(vessel_dir / vf), cv2.IMREAD_UNCHANGED)

    orig = None
    for n in TWO:
        if os.path.splitext(n)[0] == stem:
            orig = cv2.imread(str(IMG_IN / n))
            break
    if orig is None:
        print(f"{stem}: 找不到对应原图")
        continue

    h, w = orig.shape[:2]
    bounds = None
    if df_bounds is not None and id_col is not None:
        row = df_bounds[df_bounds[id_col].astype(str) == stem]
        if len(row) and len(guess) == 4:
            bounds = (row.iloc[0][guess["x"]], row.iloc[0][guess["y"]],
                      row.iloc[0][guess["w"]], row.iloc[0][guess["h"]])
            print(f"{stem}: bounds={bounds}")

    mapped = map_vessel_to_original(varr, h, w, bounds)
    print(f"{stem}: vessel={varr.shape} -> mapped={mapped.shape} ratio={mapped.mean():.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"original {w}x{h}")
    axes[1].imshow(varr, cmap="gray")
    axes[1].set_title(f"vascx vessel {varr.shape[1]}x{varr.shape[0]}")
    axes[2].imshow(mapped, cmap="gray")
    axes[2].set_title("mapped to original")
    axes[3].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    axes[3].imshow(np.ma.masked_where(mapped == 0, mapped), cmap="autumn", alpha=0.75)
    axes[3].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
''')

# ----------------------------------------------------------------------------
md(r"""
## 10. 结论怎么读

跑完之后，请对照这张表告诉我结果：

| 观察 | 含义 | 下一步 |
|---|---|---|
| 安装 returncode=0，CUDA 仍为 True，`vascx` 命令存在 | 链路通 | 进入全量集成 |
| CUDA 变 False | torch 被降级 | 需要固定 torch 版本再装 |
| 安装失败 | 依赖冲突 | 改用 `--no-deps` + 手动补依赖，或放弃路线 B |
| `run-models` returncode=0，`vessels/` 有图 | 权重下载 + 推理成功 | 看映射是否对齐 |
| `vessels/` 图是 1024×1024 | 预处理空间输出 | 必须用 bounds 映射 |
| 映射后叠加图血管和原图对得上 | 映射正确 | 可以直接用 |
| 叠加图整体偏移/缩放 | bounds 含义猜错 | 把 bounds.csv 内容发我，我推公式 |

**最需要你确认的是第 9 节最后两张叠加图**——血管是否严丝合缝地贴在原图的血管上。
如果对不上，先别继续，把 `bounds.csv` 的完整内容贴给我。
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

out = Path(r"E:\machine learing\project4\vascx_probe.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("written:", out, out.stat().st_size, "bytes,", len(cells), "cells")
