# -*- coding: utf-8 -*-
"""Run VascX one image at a time (memory safety), then merge results."""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\machine learing\project4")
VX = ROOT / "vascx_env" / "Scripts" / "vascx.exe"


def run_one(img_path: Path, work: Path):
    shutil.rmtree(work, ignore_errors=True)
    (work / "in").mkdir(parents=True)
    (work / "out").mkdir(parents=True)
    shutil.copy(img_path, work / "in" / img_path.name)

    cmd = [str(VX), "run-models", str(work / "in"), str(work / "out"),
           "--device", "cpu", "--n-jobs", "1",
           "--no-disc", "--no-fovea", "--no-quality", "--no-overlay"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = ROOT / "data" / "segmentation" / args.split / "image"
    out_root = Path(args.out) if args.out else ROOT / f"vascx_{args.split}_out"
    vessels_dir = out_root / "vessels"
    vessels_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted([f for f in src.iterdir() if f.suffix.lower() == ".jpg"],
                 key=lambda p: int(p.stem))
    work = ROOT / "vascx_tmp"

    bounds_rows = []
    t0 = time.time()
    ok, fail = 0, 0

    for i, img in enumerate(ids, 1):
        t = time.time()
        p = run_one(img, work)
        vp = work / "out" / "vessels" / f"{img.stem}.png"
        bp = work / "out" / "bounds.csv"

        if p.returncode == 0 and vp.exists():
            shutil.copy(vp, vessels_dir / vp.name)
            if bp.exists():
                bounds_rows.append(bp.read_text(encoding="utf-8").splitlines())
            ok += 1
            status = "ok"
        else:
            fail += 1
            status = "FAIL"
            tail = (p.stderr or "")[-300:].replace("\n", " ")

        print(f"[{i:02d}/{len(ids)}] {img.name:8s} {status} {time.time()-t:5.1f}s "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
        if status == "FAIL":
            print("   ", tail, flush=True)

        sys.stdout.flush()

    # merge bounds.csv
    if bounds_rows:
        merged = out_root / "bounds.csv"
        with open(merged, "w", encoding="utf-8") as f:
            f.write("id,success,bounds\n")
            for lines in bounds_rows:
                for ln in lines[1:]:
                    f.write(ln + "\n")
        print("merged bounds ->", merged)

    print()
    print(f"done: {ok} ok, {fail} failed, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
