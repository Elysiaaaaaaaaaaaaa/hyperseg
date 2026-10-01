"""Search the evaluation protocol that reproduces the v2 checkpoint's recorded val_miou.

``v2_val_replay.py`` and ``v2_metric_probe.py`` bracket the recorded
``val_miou = 0.7631595244109886`` without hitting it: at size 768 the per-batch
per-class mean gives 0.7403 while the dataset-wide confusion matrix gives 0.7941, and
the recorded value sits between them.  Two knobs move a score in exactly that
direction and both are cheap to test offline from a single pass:

* **batch size** -- ``mean_iou`` averages per-class IoU inside each batch, and a batch
  that covers more images has more classes "present", so the score rises towards the
  dataset-wide value as the batch grows (the v1 checkpoint matched at batch 2);
* **horizontal-flip TTA** -- averaging logits with their mirror image normally adds
  one to three points.

This collects one confusion matrix per image for each variant, then sweeps batch sizes
offline, so a single GPU pass answers both questions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from hyperseg_uav import UAVDataset  # noqa: E402
from v2_model import load_v2  # noqa: E402

CLASSES = 9
IGNORE = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/hyperseg_resume_best.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/low_altitude_2026')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'runs/splits')
    parser.add_argument('--sizes', type=int, nargs='+', default=[768])
    parser.add_argument('--batches', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 699])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'runs/b3_resume_test2_20260930/v2_protocol_search.json')
    return parser.parse_args()


def confusion(truth, guess):
    valid = truth != IGNORE
    return torch.bincount(truth[valid] * CLASSES + guess[valid],
                          minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES).double().cpu().numpy()


def class_iou(matrix):
    diagonal = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    present = [c for c in range(1, CLASSES) if union[c] > 0]
    if not present:
        return None
    return float((diagonal[present] / union[present]).mean())


def batch_class_mean(matrices, batch_size):
    scores = []
    for start in range(0, len(matrices), batch_size):
        score = class_iou(sum(matrices[start:start + batch_size]))
        if score is not None:
            scores.append(score)
    return sum(scores) / max(1, len(scores))


@torch.no_grad()
def collect(model, ids, data, size, device, workers):
    dataset = UAVDataset(data / 'train/images', data / 'train/masks', ids, size, False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    variants = ('plain', 'hflip', 'hflip_tta')
    collected = {name: [] for name in variants}
    started = time.monotonic()
    for index, batch in enumerate(loader, start=1):
        image = batch['image'].to(device)
        target = batch['mask'][0].to(device)
        logits = model(image)['logits']
        flipped = model(image.flip(-1))['logits'].flip(-1)
        collected['plain'].append(confusion(target, logits.argmax(1)[0]))
        collected['hflip'].append(confusion(target, flipped.argmax(1)[0]))
        collected['hflip_tta'].append(confusion(target, (0.5 * (logits + flipped)).argmax(1)[0]))
        if index % 200 == 0:
            print(f'  collected {index}/{len(dataset)}', flush=True)
    return collected, time.monotonic() - started


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids = [line.strip() for line in (args.split_dir / 'val.txt').read_text().splitlines() if line.strip()]
    model, state, _ = load_v2(args.checkpoint, device)
    recorded = state.get('val_miou')
    print(f'device={device} val_samples={len(ids)} recorded={recorded}', flush=True)

    trials, hits = [], []
    for size in args.sizes:
        collected, elapsed = collect(model, ids, args.data, size, device, args.num_workers)
        print(f'size={size}: collected {len(collected["plain"])} images in {elapsed:.0f}s', flush=True)
        size_trials = []
        for variant, matrices in collected.items():
            for batch_size in args.batches:
                if batch_size > len(matrices):
                    continue
                score = batch_class_mean(matrices, batch_size)
                difference = abs(score - recorded)
                entry = {'size': size, 'variant': variant, 'batch_size': batch_size,
                         'replayed_val_miou': score, 'absolute_difference': difference,
                         'matches': difference < args.tolerance}
                size_trials.append(entry)
                if entry['matches']:
                    hits.append(entry)
                    print(f"  HIT size={size} variant={variant} batch={batch_size} "
                          f"score={score:.6f}", flush=True)
        best = min(size_trials, key=lambda item: item['absolute_difference'])
        print(f'  closest: variant={best["variant"]} batch={best["batch_size"]} '
              f'score={best["replayed_val_miou"]:.6f} diff={best["absolute_difference"]:.2e}', flush=True)
        trials.extend(size_trials)

    summary = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': state.get('epoch'),
        'recorded_val_miou': recorded,
        'tolerance': args.tolerance,
        'note': 'hflip_tta averages raw logits with the mirror image, matching tools/infer_hyperseg.py',
        'hits': hits,
        'trials': trials,
        'device': str(device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'recorded': recorded, 'hits': hits}, indent=2), flush=True)
    print(f'PROTOCOL_FOUND={hits[0]}' if hits else 'NO_PROTOCOL_MATCHED', flush=True)


if __name__ == '__main__':
    main()
