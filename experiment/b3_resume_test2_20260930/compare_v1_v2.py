"""Compare the v2 resume-checkpoint export against the published v1 export.

Both packages cover the same 1300 round-two images at the same 768/0.5/no-TTA
protocol, so every difference is attributable to the checkpoint.  Reports the
per-class pixel share shift, the overall fraction of pixels that changed label, and
the images that moved the most -- enough to judge whether the continued-training
weights actually behave differently from the submitted ones.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / 'experiment/b3_test2_20260929/hyperseg_b3_test2_predictions.zip'
V2 = ROOT / 'experiment/b3_resume_test2_20260930/hyperseg_b3_resume_test2_predictions.zip'
OUT = ROOT / 'experiment/b3_resume_test2_20260930/logs/v1_v2_comparison.json'
CLASS_NAMES = {1: 'Background', 2: 'Building', 3: 'Road', 4: 'Water',
               5: 'Barren', 6: 'Vegetation', 7: 'Agricultural', 8: 'Vehicle'}


def load(path: Path):
    with zipfile.ZipFile(path) as archive:
        return {name: np.asarray(Image.open(io.BytesIO(archive.read(name))))
                for name in sorted(archive.namelist())}


def main():
    first, second = load(V1), load(V2)
    if first.keys() != second.keys():
        raise SystemExit('the two packages do not cover the same images')

    counts_v1 = np.zeros(9, dtype=np.int64)
    counts_v2 = np.zeros(9, dtype=np.int64)
    changed_pixels = total_pixels = 0
    per_image = []

    for name in first:
        old, new = first[name], second[name]
        counts_v1 += np.bincount(old.ravel(), minlength=9)
        counts_v2 += np.bincount(new.ravel(), minlength=9)
        changed = int((old != new).sum())
        changed_pixels += changed
        total_pixels += old.size
        per_image.append({'name': name, 'changed_pixels': changed,
                          'changed_fraction': changed / old.size})

    per_image.sort(key=lambda item: item['changed_pixels'], reverse=True)
    fraction = {cls: float(counts_v2[cls] / counts_v1[cls]) if counts_v1[cls] else None
                for cls in range(1, 9)}
    summary = {
        'v1_package': str(V1.relative_to(ROOT)),
        'v2_package': str(V2.relative_to(ROOT)),
        'images': len(first),
        'class_share_v1': {str(c): float(counts_v1[c]) / counts_v1[1:].sum() for c in range(1, 9)},
        'class_share_v2': {str(c): float(counts_v2[c]) / counts_v2[1:].sum() for c in range(1, 9)},
        'pixel_count_ratio_v2_over_v1': {str(c): fraction[c] for c in range(1, 9)},
        'class_names': {str(c): CLASS_NAMES[c] for c in range(1, 9)},
        'changed_pixel_fraction': changed_pixels / total_pixels,
        'images_with_any_change': sum(1 for item in per_image if item['changed_pixels']),
        'mean_changed_fraction': sum(item['changed_fraction'] for item in per_image) / len(per_image),
        'top_changed_images': per_image[:15],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + '\n')

    print(f"changed pixels overall: {summary['changed_pixel_fraction']:.4%}")
    print(f"images with any change: {summary['images_with_any_change']}/{len(first)}")
    print(f"{'cls':>3} {'name':<12} {'v1 share':>9} {'v2 share':>9} {'v2/v1':>7}")
    for cls in range(1, 9):
        print(f"{cls:>3} {CLASS_NAMES[cls]:<12} "
              f"{summary['class_share_v1'][str(cls)]:>9.4%} "
              f"{summary['class_share_v2'][str(cls)]:>9.4%} "
              f"{fraction[cls]:>7.3f}")
    print('most-changed images:')
    for item in per_image[:10]:
        print(f"  {item['name']:<18} {item['changed_pixels']:>7} px "
              f"({item['changed_fraction']:.2%})")
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
