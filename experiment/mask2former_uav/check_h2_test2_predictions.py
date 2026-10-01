"""Verify an H2 (Swin-L + Mask2Former) test_2 prediction export.

Two questions are answered, and they are deliberately kept separate:

* **Format** - does every file satisfy the competition manual, section 13?
  The raw PNG chunks are read rather than Pillow's ``mode``, so a palette
  (PLTE) or transparency (tRNS) chunk cannot slip through, and the file-name
  set is compared against the unlabelled input exactly.
* **Behaviour** - the interesting failure mode on this dataset is a prediction
  that is format-correct yet has collapsed onto one or two classes.  For that
  the script reports the global pixel share per class id and, per image, how
  many classes appear and which one dominates.

Exported ids live in the *raw* label space (1..8, 0 = Ignore): the training
pipeline sets ``reduce_zero_label=True`` and MMSeg shifts the ids back on
export, so 1 = Background, 2 = Building, ... 8 = Vehicle.

    python experiment/mask2former_uav/check_h2_test2_predictions.py \
        --predictions experiment/mask2former_uav/results/h2_160k_best132k_test2/pred_test2_h2_best132k \
        --source dataset/low_altitude_2026/test_2/images \
        --label best132k
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = {
    1: 'Background', 2: 'Building', 3: 'Road', 4: 'Water',
    5: 'Barren', 6: 'Vegetation', 7: 'Agricultural', 8: 'Vehicle',
}
EXPECTED_SIDE = (1024, 1024)
VALID_IDS = set(range(9))


def png_chunks(data: bytes):
    """Yield (type, payload) for every chunk in a PNG byte string."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('missing PNG signature')
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8].decode('ascii', 'replace')
        payload = data[offset + 8:offset + 8 + length]
        yield chunk_type, payload
        offset += 12 + length
        if chunk_type == 'IEND':
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--label', default='export')
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--bench', type=Path, default=None,
                        help='Optional second predictions dir to compare shares against')
    return parser.parse_args()


def scan(predictions: Path, source: Path) -> dict:
    report: dict = {
        'label': predictions.name,
        'predictions': str(predictions),
        'source': str(source),
    }
    pred_files = sorted(predictions.glob('*.png'))
    source_names = {p.name for p in source.glob('*.png')}
    pred_names = {p.name for p in pred_files}

    report['entries'] = len(pred_files)
    report['source_entries'] = len(source_names)
    report['non_png_entries'] = [
        p.name for p in predictions.iterdir()
        if p.is_file() and not p.suffix.lower() == '.png']
    report['name_set_matches_source'] = pred_names == source_names
    report['missing_names'] = sorted(source_names - pred_names)[:5]
    report['extra_names'] = sorted(pred_names - source_names)[:5]
    report['missing_count'] = len(source_names - pred_names)
    report['extra_count'] = len(pred_names - source_names)

    failures: list[str] = []
    counts: collections.Counter = collections.Counter()
    histogram: collections.Counter = collections.Counter()
    dominant: collections.Counter = collections.Counter()
    class_count_hist: collections.Counter = collections.Counter()
    collapsed_single: list[str] = []
    byte_total = 0
    digest = hashlib.sha256()

    for path in pred_files:
        data = path.read_bytes()
        byte_total += len(data)
        digest.update(data)
        counts['files'] += 1
        name = path.name

        chunk_types = []
        ihdr = None
        for chunk_type, payload in png_chunks(data):
            chunk_types.append(chunk_type)
            if chunk_type == 'IHDR':
                ihdr = struct.unpack('>IIBBBBB', payload)
        if ihdr is None:
            failures.append(f'{name}: no IHDR chunk')
            continue
        width, height, depth, color_type, _comp, _filt, interlace = ihdr

        if color_type != 0:
            counts[f'color_type_{color_type}'] += 1
            failures.append(
                f'{name}: color type {color_type} (not single-channel gray)')
        else:
            counts['color_type_gray'] += 1
        if depth != 8:
            counts[f'bit_depth_{depth}'] += 1
            failures.append(f'{name}: bit depth {depth}, expected 8')
        if 'PLTE' in chunk_types:
            counts['with_plte_chunk'] += 1
            failures.append(f'{name}: contains PLTE chunk')
        if 'tRNS' in chunk_types:
            counts['with_trns_chunk'] += 1
            failures.append(f'{name}: contains tRNS chunk')
        if interlace != 0:
            failures.append(f'{name}: interlaced PNG')
        if (width, height) != EXPECTED_SIDE:
            failures.append(f'{name}: size {width}x{height}, expected 1024x1024')

        with Image.open(io.BytesIO(data)) as image:
            if image.mode != 'L':
                failures.append(f'{name}: Pillow mode {image.mode}, expected L')
            if image.size != EXPECTED_SIDE:
                failures.append(f'{name}: Pillow size {image.size}')
            pixels = image.tobytes()
            if len(pixels) != EXPECTED_SIDE[0] * EXPECTED_SIDE[1]:
                failures.append(
                    f'{name}: {len(pixels)} pixels, expected 1048576')
            values = set(pixels)
            if not values <= VALID_IDS:
                failures.append(
                    f'{name}: class ids outside 0..8 -> {sorted(values)[:8]}')
            per_image = collections.Counter(pixels)
            histogram.update(per_image)

        class_count_hist[len(per_image)] += 1
        top_id, top_n = per_image.most_common(1)[0]
        dominant[top_id] += 1
        if len(per_image) == 1:
            collapsed_single.append(name)

    total_pixels = sum(histogram.values()) or 1
    report['counts'] = dict(counts)
    report['failures'] = failures[:20]
    report['failure_count'] = len(failures)
    report['bytes'] = byte_total
    report['sha256'] = digest.hexdigest()
    report['class_ids_present'] = sorted(histogram)
    report['pixel_share'] = {
        str(k): round(histogram[k] / total_pixels, 6) for k in sorted(histogram)}
    report['pixel_share_named'] = {
        CLASS_NAMES.get(k, f'id{k}'): round(histogram[k] / total_pixels, 6)
        for k in sorted(histogram) if k in CLASS_NAMES}
    report['background_share'] = round(histogram.get(1, 0) / total_pixels, 6)
    report['classes_per_image_hist'] = {
        str(k): v for k, v in sorted(class_count_hist.items())}
    report['dominant_class_hist'] = {
        CLASS_NAMES.get(k, f'id{k}'): v for k, v in dominant.most_common()}
    report['single_class_images'] = len(collapsed_single)
    report['single_class_examples'] = collapsed_single[:5]

    report['verdict'] = (
        'PASS' if not failures
        and report['name_set_matches_source']
        and report['entries'] == report['source_entries']
        else 'FAIL')
    return report


def compare(report: dict, bench_path: Path) -> dict:
    """Side-by-side pixel share against another export (direction only)."""
    with open(bench_path, encoding='utf-8') as handle:
        bench = json.load(handle)
    rows = []
    for key in sorted(CLASS_NAMES.values(), key=lambda n: list(CLASS_NAMES.values()).index(n)):
        now = report['pixel_share_named'].get(key, 0.0)
        before = bench.get('pixel_share_named', {}).get(key, 0.0)
        rows.append({'class': key, 'share': now, 'reference': before,
                     'delta': round(now - before, 6)})
    return {'reference_label': bench.get('label'), 'rows': rows,
            'background_share': report['background_share'],
            'reference_background_share': bench.get('background_share')}


def main() -> None:
    args = parse_args()
    if not args.predictions.is_dir():
        raise SystemExit(f'not a directory: {args.predictions}')
    if not args.source.is_dir():
        raise SystemExit(f'not a directory: {args.source}')

    report = scan(args.predictions, args.source)
    report['label'] = args.label
    if args.bench and args.bench.is_file():
        report['comparison'] = compare(report, args.bench)

    out = args.report or (
        args.predictions.parent / f'check_{args.label}.json')
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    printable = {k: v for k, v in report.items() if k != 'comparison'}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    if 'comparison' in report:
        print('comparison vs', report['comparison']['reference_label'])
        for row in report['comparison']['rows']:
            print(f"  {row['class']:<13} {row['share']:.4f} "
                  f"(ref {row['reference']:.4f}, {row['delta']:+.4f})")
    print('report ->', out)
    sys.exit(0 if report['verdict'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
