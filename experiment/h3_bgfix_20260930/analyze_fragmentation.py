# -*- coding: utf-8 -*-
"""掩码碎裂与微小车辆误检的定量归因分析（纯 CPU，读四臂 test_2 预测 + 训练真值参照）。

指标定义：
- boundary_density: 4 邻域内标签不同的像素对占比（忽略 0 不参与），越大越碎。
- 对类 {1,2,3,7,8} 做连通域分析：组件数、组件面积中位数、
  类内像素落在 <64 / <256 像素小组件中的份额（"碎屑份额"）。
- Vehicle(8) 专项：每图车辆像素数、组件数、小组件份额。
参照：训练真值掩码的同指标（"自然碎裂"水平）。
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = Path(__file__).resolve().parent / "results" / "predictions"
GT_DIR = ROOT / "dataset" / "low_altitude_2026" / "train" / "masks"

SETS = {
    "baseline": PRED_DIR / "pred_test2_baseline_20k",
    "aug": PRED_DIR / "pred_test2_aug_20k",
    "loss": PRED_DIR / "pred_test2_loss_20k",
    "full": PRED_DIR / "pred_test2_full_20k",
}
CC_CLASSES = [1, 2, 3, 7, 8]
SMALL1, SMALL2 = 64, 256


def boundary_density(mask: np.ndarray) -> float:
    valid = mask != 0
    diff_h = (mask[:, 1:] != mask[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    diff_v = (mask[1:, :] != mask[:-1, :]) & valid[1:, :] & valid[:-1, :]
    tot = valid[:, 1:].sum() + valid[1:, :].sum()
    if tot == 0:
        return 0.0
    return float((diff_h.sum() + diff_v.sum()) / tot)


def cc_stats(mask: np.ndarray, cls: int):
    m = mask == cls
    n_px = int(m.sum())
    if n_px == 0:
        return dict(px=0, n_comp=0, med_size=0.0, small1_share=0.0, small2_share=0.0)
    lab, n = ndimage.label(m)
    sizes = np.bincount(lab.ravel())[1:]
    small1 = int(sizes[sizes < SMALL1].sum())
    small2 = int(sizes[sizes < SMALL2].sum())
    return dict(
        px=n_px,
        n_comp=int(n),
        med_size=float(np.median(sizes)),
        small1_share=small1 / n_px,
        small2_share=small2 / n_px,
    )


def analyze_set(name: str, files, out: dict):
    agg_bd = []
    per_cls = {c: dict(px=0, n_comp=0, sizes=[], small1=0, small2=0) for c in CC_CLASSES}
    veh_img_px = []  # 每图车辆像素（含 0）
    for f in files:
        mask = np.array(Image.open(f))
        agg_bd.append(boundary_density(mask))
        veh_img_px.append(int((mask == 8).sum()))
        for c in CC_CLASSES:
            s = cc_stats(mask, c)
            d = per_cls[c]
            d["px"] += s["px"]
            d["n_comp"] += s["n_comp"]
            d["small1"] += s["small1_share"] * s["px"]
            d["small2"] += s["small2_share"] * s["px"]
            if s["px"] > 0:
                d["sizes"].append(s["med_size"])
    res = {
        "n_images": len(files),
        "boundary_density_mean": float(np.mean(agg_bd)),
        "boundary_density_p90": float(np.percentile(agg_bd, 90)),
        "vehicle": {
            "img_with_vehicle": int(sum(1 for v in veh_img_px if v > 0)),
            "mean_px_per_img": float(np.mean(veh_img_px)),
        },
        "classes": {},
    }
    for c, d in per_cls.items():
        if d["px"] == 0:
            continue
        res["classes"][str(c)] = {
            "px_share": d["px"] / (len(files) * 1024 * 1024),
            "comp_per_mpx": d["n_comp"] / (d["px"] / 1e6),
            "med_comp_size": float(np.median(d["sizes"])) if d["sizes"] else 0.0,
            "small1_share": d["small1"] / d["px"],
            "small2_share": d["small2"] / d["px"],
        }
    out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--gt-sample", type=int, default=400)
    ap.add_argument("--output", default=str(Path(__file__).parent / "results" / "fragmentation_20k.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = {}
    for name, d in SETS.items():
        files = sorted(d.glob("*.png"))
        if len(files) > args.sample:
            files = rng.sample(files, args.sample)
        print(f"[{name}] {len(files)} images ...", flush=True)
        analyze_set(name, files, out)

    gt_files = sorted(GT_DIR.glob("*.png"))
    if len(gt_files) > args.gt_sample:
        gt_files = rng.sample(gt_files, args.gt_sample)
    print(f"[train_gt] {len(gt_files)} masks ...", flush=True)
    analyze_set("train_gt", gt_files, out)

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"saved -> {args.output}")

    # 控制台速览表
    hdr = f"{'set':<10} {'bd_mean':>8} {'bd_p90':>8}"
    for c in CC_CLASSES:
        hdr += f" | c{c}: comp/mpx  med  s<{SMALL1}  s<{SMALL2}"
    print(hdr)
    for name, r in out.items():
        line = f"{name:<10} {r['boundary_density_mean']:>8.4f} {r['boundary_density_p90']:>8.4f}"
        for c in CC_CLASSES:
            cl = r["classes"].get(str(c))
            if cl:
                line += f" | {cl['comp_per_mpx']:>7.1f} {cl['med_comp_size']:>6.0f} {cl['small1_share']:>6.3f} {cl['small2_share']:>6.3f}"
            else:
                line += " |      -"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
