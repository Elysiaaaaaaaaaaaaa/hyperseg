#!/usr/bin/env python
"""判别「低细节」是病因，还是「外观偏移」的替身。

在 test_2 上给每张图补算逐通道均值 / 亮度 / 色偏，然后看：
  * detail_lap 与外观量（亮度、色偏）的相关 —— 若强，说明低细节图同时是暗/偏色图，
    「细节尺度」只是外观偏移的一个共线指标；
  * 预测背景占比与各量的相关 —— 谁的相关更强，谁更接近真正的自变量。

用法：
  python experiment/mask2former_uav/attribute_detail_vs_appearance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / 'experiment/mask2former_uav/results/stats_test2_vs_predbg.json'
IMAGES = ROOT / 'dataset/low_altitude_2026/test_2/images'


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    d = json.loads(STATS.read_text(encoding='utf-8'))
    rows = d['rows']
    for i, r in enumerate(rows):
        a = np.asarray(Image.open(IMAGES / f"{r['stem']}.png").convert('RGB'),
                       dtype=np.float32)
        m = a.reshape(-1, 3).mean(0)
        r['mean_r'], r['mean_g'], r['mean_b'] = (float(m[0]), float(m[1]), float(m[2]))
        r['lum'] = 0.299 * float(m[0]) + 0.587 * float(m[1]) + 0.114 * float(m[2])
        r['cast_gr'] = float(m[1] - m[0])
        r['cast_br'] = float(m[2] - m[0])
        if (i + 1) % 400 == 0:
            print(f'  ... {i + 1}/{len(rows)}')

    pred = [r['h2_best132k__pred_bg_frac'] for r in rows]

    print('\n--- detail_lap 与外观量的相关（test_2, %d 张）---' % len(rows))
    for k in ['lum', 'cast_gr', 'cast_br', 'mean_r', 'contrast_std']:
        print(f'  detail_lap ~ {k:12s} rho={spearman([r["detail_lap"] for r in rows], [r[k] for r in rows]):+.3f}')

    print('\n--- 预测背景占比（h2 best132k）与各量的相关 ---')
    for k in ['detail_lap', 'detail_hf', 'lum', 'cast_gr', 'cast_br',
              'mean_r', 'mean_b', 'contrast_std']:
        print(f'  pred_bg    ~ {k:12s} rho={spearman([r[k] for r in rows], pred):+.3f}')

    print('\n--- 参考：训练集外观均值 R/G/B = 106.3 / 107.8 / 100.2 ---')
    for k in ['mean_r', 'mean_g', 'mean_b', 'lum', 'cast_gr']:
        v = np.array([r[k] for r in rows], float)
        print(f'  test_2 {k:8s} mean={v.mean():8.2f}  p10={np.percentile(v,10):8.2f}  p90={np.percentile(v,90):8.2f}')

    STATS.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'\nappended channel means -> {STATS}')


if __name__ == '__main__':
    main()
