"""Find the evaluation resolution that reproduces the v2 checkpoint's recorded val_miou.

``models/hyperseg_resume_best.pt`` records ``val_miou = 0.7631595244109886`` but ships
no note of the resolution it was measured at, and the training script for the v2
architecture is not in the repository.  The v1 checkpoint's 0.7420226665631505
reproduces exactly at size 768 / batch 2 under ``hyperseg_uav.losses.mean_iou``
averaged over batches (see ``val_sanity.py``), so the same protocol is replayed here
at several resolutions.  Whichever one matches tells us the scale the test-set export
must use: inference has to run at the resolution the checkpoint was selected on,
otherwise the sliding-window export silently disagrees with validation.
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

from hyperseg_uav import UAVDataset, mean_iou  # noqa: E402
from v2_model import load_v2  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/hyperseg_resume_best.pt')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/low_altitude_2026')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'runs/splits')
    parser.add_argument('--sizes', type=int, nargs='+', default=[768, 512, 1024])
    parser.add_argument('--normalize', nargs='+', choices=('none', 'imagenet'), default=['none'],
                        help="none = the v1 convention (divide by 255 only); "
                             "imagenet = Hugging Face SegformerImageProcessor statistics")
    parser.add_argument('--stop-on-match', action='store_true',
                        help='stop the grid at the first matching combination')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'runs/b3_resume_test2_20260930/v2_resolution_replay.json')
    return parser.parse_args()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def replay(model, ids, data, size, batch_size, device, workers, normalize='none'):
    dataset = UAVDataset(data / 'train/images', data / 'train/masks', ids, size, False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    if normalize == 'imagenet':
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    else:
        mean = std = None
    score_sum, batch_count = 0.0, 0
    started = time.monotonic()
    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        target = batch['mask'].to(device, non_blocking=True)
        if mean is not None:
            image = (image - mean) / std
        score_sum += mean_iou(model(image)['logits'], target)
        batch_count += 1
    return {
        'size': size,
        'normalize': normalize,
        'samples': len(dataset),
        'batches': batch_count,
        'replayed_val_miou': score_sum / max(1, batch_count),
        'elapsed_seconds': time.monotonic() - started,
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids = [line.strip() for line in (args.split_dir / 'val.txt').read_text().splitlines() if line.strip()]
    print(f'device={device} val_samples={len(ids)}', flush=True)

    model, state, info = load_v2(args.checkpoint, device)
    print(json.dumps(info, indent=2), flush=True)
    recorded = state.get('val_miou')

    trials, matched = [], None
    for normalize in args.normalize:
        for size in args.sizes:
            result = replay(model, ids, args.data, size, args.batch_size, device,
                            args.num_workers, normalize)
            result['recorded_val_miou'] = recorded
            result['absolute_difference'] = abs(result['replayed_val_miou'] - recorded)
            result['matches'] = result['absolute_difference'] < args.tolerance
            trials.append(result)
            print(f"normalize={normalize:9s} size={size:5d} "
                  f"replayed={result['replayed_val_miou']:.6f} "
                  f"diff={result['absolute_difference']:.2e} "
                  f"({'MATCH' if result['matches'] else 'no'}) "
                  f"in {result['elapsed_seconds']:.0f}s", flush=True)
            if result['matches'] and matched is None:
                matched = {'size': size, 'normalize': normalize}
                if args.stop_on_match:
                    break
        if matched and args.stop_on_match:
            break

    summary = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': state.get('epoch'),
        'recorded_val_miou': recorded,
        'tolerance': args.tolerance,
        'protocol': 'size = square resize, batch 2, hyperseg_uav.losses.mean_iou averaged over batches',
        'trials': trials,
        'matched': matched,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'MATCHED={matched}' if matched else 'NO_COMBINATION_MATCHED', flush=True)


if __name__ == '__main__':
    main()
