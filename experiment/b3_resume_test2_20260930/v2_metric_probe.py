"""Identify the metric definition behind the v2 checkpoint's recorded val_miou.

``models/hyperseg_resume_best.pt`` records ``val_miou = 0.7631595244109886``, but the
v1 definition used by ``hyperseg_uav.losses.mean_iou`` (per-class IoU within a batch,
averaged over batches) tops out at 0.7426 on this split -- across sizes 512/768/1024
and both preprocessing conventions (see ``v2_val_replay.py``).  Since the v2 training
script is not in the repository, the remaining variable is the metric itself, so this
probe computes several common definitions from a single forward pass:

* ``batch_class_mean``   -- per-class IoU inside each batch, averaged over batches (v1)
* ``dataset_class_mean`` -- per-class IoU from one dataset-wide confusion matrix
* ``image_class_mean``   -- per-class IoU inside each image, averaged over images
* ``dataset_class_mean_with_ignore`` -- as above but averaging classes 0..8

If one of these lands on the recorded value, the evaluation protocol is identified and
the export resolution can be trusted; if none does, the number is simply not
reproducible from the current repository and must be reported as such.
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
    parser.add_argument('--sizes', type=int, nargs='+', default=[768, 1024])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'runs/b3_resume_test2_20260930/v2_metric_probe.json')
    return parser.parse_args()


def class_iou(confusion: torch.Tensor, classes=(1, 2, 3, 4, 5, 6, 7, 8)):
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = [c for c in classes if union[c] > 0]
    if not present:
        return None
    return float((intersection[present] / union[present]).mean())


@torch.no_grad()
def probe(model, ids, data, size, batch_size, device, workers):
    dataset = UAVDataset(data / 'train/images', data / 'train/masks', ids, size, False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == 'cuda')

    total = torch.zeros(CLASSES, CLASSES, dtype=torch.float64)
    batch_scores, image_scores = [], []
    started = time.monotonic()

    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        target = batch['mask'].to(device, non_blocking=True)
        prediction = model(image)['logits'].argmax(1)
        batch_confusion = torch.zeros(CLASSES, CLASSES, dtype=torch.float64, device=device)
        for index in range(target.shape[0]):
            truth, guess = target[index], prediction[index]
            valid = truth != IGNORE
            flat = torch.bincount(truth[valid] * CLASSES + guess[valid],
                                  minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES).double()
            total += flat.cpu()
            batch_confusion += flat
            score = class_iou(flat)
            if score is not None:
                image_scores.append(score)
        score = class_iou(batch_confusion)
        if score is not None:
            batch_scores.append(score)

    return {
        'size': size,
        'samples': len(dataset),
        'batches': len(batch_scores),
        'batch_class_mean': sum(batch_scores) / max(1, len(batch_scores)),
        'dataset_class_mean': class_iou(total),
        'image_class_mean': sum(image_scores) / max(1, len(image_scores)),
        'dataset_class_mean_with_ignore': class_iou(total, classes=tuple(range(CLASSES))),
        'elapsed_seconds': time.monotonic() - started,
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids = [line.strip() for line in (args.split_dir / 'val.txt').read_text().splitlines() if line.strip()]

    model, state, _ = load_v2(args.checkpoint, device)
    recorded = state.get('val_miou')
    print(f'device={device} val_samples={len(ids)} recorded_val_miou={recorded}', flush=True)

    trials = []
    for size in args.sizes:
        result = probe(model, ids, args.data, size, args.batch_size, device, args.num_workers)
        result['recorded_val_miou'] = recorded
        result['candidates'] = {
            name: {'value': result[name],
                   'absolute_difference': abs(result[name] - recorded),
                   'matches': abs(result[name] - recorded) < 1e-3}
            for name in ('batch_class_mean', 'dataset_class_mean', 'image_class_mean',
                         'dataset_class_mean_with_ignore')
        }
        result['matching_definitions'] = [name for name, entry in result['candidates'].items()
                                          if entry['matches']]
        trials.append(result)
        print(f"size={size:5d} batch={result['batch_class_mean']:.6f} "
              f"dataset={result['dataset_class_mean']:.6f} "
              f"image={result['image_class_mean']:.6f} "
              f"with_ignore={result['dataset_class_mean_with_ignore']:.6f} "
              f"hits={result['matching_definitions']} in {result['elapsed_seconds']:.0f}s", flush=True)

    summary = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': state.get('epoch'),
        'recorded_val_miou': recorded,
        'trials': trials,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
