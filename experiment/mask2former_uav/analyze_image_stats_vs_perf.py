#!/usr/bin/env python
"""把「图像本身的对比度 / 细节尺度」与「预测质量」做分层统计。

回答的问题：
  低对比度、低分辨率的图，是不是真的被预测得更差？这是"图像难"还是"模型偏置"？

三条证据链：
  A. test_2（1300 张无标注）：逐图图像统计 vs 各臂的逐图背景占比（塌陷度量）
     —— 检验"低对比度 ⇒ 被塌陷成背景"是否成立。
  B. test.txt（699 张有标注）：逐图图像统计 vs 逐图 GT 背景占比
     —— 检验混淆项："低对比度的图在真值里本来就以背景为主"。
  C. train（6996 张）：同样的分层，看训练先验是否就是这个形状。
     —— 若先验如此，模型的背景偏置就是从数据里学来的，而不是缺陷。

用法：
  python experiment/mask2former_uav/analyze_image_stats_vs_perf.py \
      --images dataset/low_altitude_2026/train/images \
      --masks  dataset/low_altitude_2026/train/masks \
      --split  runs/splits/test.txt \
      --label  test_labeled \
      --stats-out experiment/mask2former_uav/results/stats_test_labeled.json

  # 无标注集 + 多臂预测
  python experiment/mask2former_uav/analyze_image_stats_vs_perf.py \
      --images dataset/low_altitude_2026/test_2/images \
      --pred "h2_20k=.../pred_test2_h2_20k" \
      --pred "h2_best132k=.../pred_test2_h2_best132k" \
      --label test2_unlabeled --stats-out .../stats_test2.json

单图统计量（灰度 c = 0.299R+0.587G+0.114B，归一化到 0..1）：
  contrast_std   全局亮度标准差（最常用的对比度度量）
  contrast_rms   标准差/均值（相对对比度，抗整体明暗影响）
  contrast_p95p5 95 分位 - 5 分位（抗离群点，反映动态范围利用）
  detail_lap     Laplacian 方差（清晰度 / 高频能量）
  detail_hf      与 sigma=2 高斯平滑之差的 std（中高频能量，抗噪）
  entropy        256 级灰度直方图熵（信息量）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def image_stats(path: Path) -> dict:
    im = Image.open(path).convert('RGB')
    a = np.asarray(im, dtype=np.float32) / 255.0
    gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    mean = float(gray.mean())
    std = float(gray.std())
    p5, p95 = np.percentile(gray, [5, 95])

    lap = ndimage.laplace(gray)
    hf = gray - ndimage.gaussian_filter(gray, sigma=2.0)

    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    return {
        'contrast_std': std,
        'contrast_rms': std / mean if mean > 1e-6 else 0.0,
        'contrast_p95p5': float(p95 - p5),
        'detail_lap': float(lap.var()),
        'detail_hf': float(hf.std()),
        'entropy': entropy,
        'mean_lum': mean,
    }


def mask_stats(path: Path) -> dict:
    m = np.asarray(Image.open(path))
    total = m.size
    vals, counts = np.unique(m, return_counts=True)
    share = {int(v): int(c) / total for v, c in zip(vals, counts)}
    return {
        'ignore_frac': share.get(0, 0.0),
        'bg_frac': share.get(1, 0.0),            # 1 = Background
        'dominant_frac': max(share.values()) if share else 0.0,
        'class_share': {str(k): v for k, v in sorted(share.items())},
    }


def pred_stats(path: Path) -> dict:
    m = np.asarray(Image.open(path))
    total = m.size
    vals, counts = np.unique(m, return_counts=True)
    share = {int(v): int(c) / total for v, c in zip(vals, counts)}
    return {
        'pred_bg_frac': share.get(1, 0.0),
        'pred_dominant_frac': max(share.values()) if share else 0.0,
        'pred_n_classes': len(share),
    }


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float('nan')


def quintile_table(values, targets: dict, n=5):
    """按 values 分位分箱，报告各箱的 target 均值。"""
    v = np.asarray(values, dtype=np.float64)
    edges = np.percentile(v, np.linspace(0, 100, n + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(v, edges[1:-1]), 0, n - 1)
    out = []
    for b in range(n):
        m = idx == b
        if not m.any():
            continue
        row = {'bin': b + 1, 'n': int(m.sum()),
               'stat_lo': float(v[m].min()), 'stat_hi': float(v[m].max()),
               'stat_mean': float(v[m].mean())}
        for name, t in targets.items():
            row[name] = float(np.asarray(t, dtype=np.float64)[m].mean())
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--masks', type=Path, default=None,
                    help='掩码目录（有标注集才给），掩码文件名与图像同名')
    ap.add_argument('--split', type=Path, default=None,
                    help='只分析该划分文件里的 stem；不给则分析整个目录')
    ap.add_argument('--pred', action='append', default=[],
                    help='name=pred_dir，可重复；目录里是同名 png')
    ap.add_argument('--label', required=True)
    ap.add_argument('--stats-out', required=True, type=Path)
    ap.add_argument('--limit', type=int, default=0, help='>0 时只跑前 N 张（调试）')
    args = ap.parse_args()

    if args.split:
        stems = [ln.strip() for ln in args.split.read_text(encoding='utf-8').splitlines()
                 if ln.strip()]
    else:
        stems = sorted(p.stem for p in args.images.glob('*.png'))
    if args.limit:
        stems = stems[:args.limit]

    pred_dirs = {}
    for spec in args.pred:
        name, _, d = spec.partition('=')
        pred_dirs[name] = Path(d)

    rows = []
    for i, stem in enumerate(stems):
        img = args.images / f'{stem}.png'
        if not img.exists():
            print(f'  [skip] missing image {img}')
            continue
        rec = {'stem': stem, **image_stats(img)}
        if args.masks is not None:
            mp = args.masks / f'{stem}.png'
            if mp.exists():
                rec.update(mask_stats(mp))
        for name, d in pred_dirs.items():
            pp = d / f'{stem}.png'
            if pp.exists():
                rec.update({f'{name}__{k}': v for k, v in pred_stats(pp).items()})
        rows.append(rec)
        if (i + 1) % 500 == 0:
            print(f'  ... {i + 1}/{len(stems)}')

    print(f'\n=== {args.label}: {len(rows)} images analysed ===')
    stat_keys = ['contrast_std', 'contrast_rms', 'contrast_p95p5',
                 'detail_lap', 'detail_hf', 'entropy']
    target_keys = [k for k in rows[0] if k.endswith('__pred_bg_frac')]
    if 'bg_frac' in rows[0]:
        target_keys = ['bg_frac'] + target_keys

    print('\n-- Spearman(图像统计, 目标) --')
    corr = {}
    for sk in stat_keys:
        for tk in target_keys:
            r = spearman([row[sk] for row in rows], [row[tk] for row in rows])
            corr[f'{sk}~{tk}'] = r
            print(f'  {sk:16s} ~ {tk:28s} rho={r:+.3f}')

    print(f'\n-- 按 {stat_keys} 五分位分箱的均值 --')
    tables = {}
    for sk in stat_keys:
        t = quintile_table([row[sk] for row in rows],
                           {tk: [row[tk] for row in rows] for tk in target_keys})
        tables[sk] = t
        print(f'  [{sk}]')
        head = f'    {"bin":>3} {"n":>6} {"stat_mean":>10} ' + \
               ' '.join(f'{tk[:14]:>15}' for tk in target_keys)
        print(head)
        for r in t:
            print(f'    {r["bin"]:>3} {r["n"]:>6} {r["stat_mean"]:>10.4f} ' +
                  ' '.join(f'{r[tk]:>15.4f}' for tk in target_keys))

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'label': args.label, 'n': len(rows),
               'stat_keys': stat_keys, 'target_keys': target_keys,
               'spearman': corr, 'quintiles': tables, 'rows': rows}
    args.stats_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                              encoding='utf-8')
    print(f'\nwrote {args.stats_out}')


if __name__ == '__main__':
    main()
