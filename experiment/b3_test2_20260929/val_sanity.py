"""Reproduce the checkpoint's recorded validation mIoU before exporting round two.

The checkpoint ``models/hyperseg_b3_best.pt`` stores ``val_miou`` computed at the
end of training with ``tools/train_hyperseg.py`` defaults (size 768, batch 2,
``hyperseg_uav.losses.mean_iou`` averaged over batches). This script replays that
exact evaluation on the fixed ``runs/splits/val.txt`` split using the same
``mean_iou`` definition, so a close match proves the Transformers-5 to
Transformers-4 key translation restores the trained weights correctly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hyperseg_uav import HyperSegUAV, UAVDataset, mean_iou, translate_legacy_keys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/low_altitude_2026')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'runs/splits')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models/hyperseg_b3_best.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/b3_test2_20260929/val_sanity.json')
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids = [line.strip() for line in (args.split_dir / 'val.txt').read_text().splitlines() if line.strip()]
    dataset = UAVDataset(args.data / 'train/images', args.data / 'train/masks', ids, args.size, False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = dict(state.get('model_config', {}))
    config['pretrained'] = False
    model = HyperSegUAV(**config).to(device)
    state_dict, translated = translate_legacy_keys(state['model'])
    model.load_state_dict(state_dict)
    model.eval()
    print(f'translated {translated} legacy encoder keys; checkpoint epoch={state.get("epoch")} '
          f'val_miou={state.get("val_miou")}', flush=True)

    started = time.monotonic()
    score_sum, batch_count, images = 0.0, 0, 0
    for batch in loader:
        image = batch['image'].to(device, non_blocking=True)
        target = batch['mask'].to(device, non_blocking=True)
        score_sum += mean_iou(model(image)['logits'], target)
        batch_count += 1
        images += image.shape[0]
        if batch_count % 50 == 0:
            print(f'progress: {images}/{len(dataset)} | running mean_iou={score_sum / batch_count:.4f}', flush=True)
    replayed = score_sum / max(1, batch_count)
    recorded = state.get('val_miou')
    result = {
        'checkpoint': str(args.checkpoint),
        'checkpoint_epoch': state.get('epoch'),
        'recorded_val_miou': recorded,
        'replayed_val_miou': replayed,
        'absolute_difference': abs(replayed - recorded) if isinstance(recorded, float) else None,
        'samples': len(dataset),
        'size': args.size,
        'batch_size': args.batch_size,
        'translated_encoder_keys': translated,
        'elapsed_seconds': time.monotonic() - started,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
