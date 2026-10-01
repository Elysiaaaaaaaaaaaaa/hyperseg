"""Export round-two test-set predictions with the HyperSeg-UAV v2 resume checkpoint.

The delivery contract is identical to ``experiment/b3_test2_20260929/run_inference.py``:
one 1024x1024 single-channel PNG per input, same filenames, packed into a zip and
checked by ``tools/check_submission.py``.

The model is different.  ``models/hyperseg_resume_best.pt`` is HyperSeg-UAV v2
(``tools/model_1.py``), so the export cannot go through ``tools/infer_hyperseg.py``,
which builds the v1 ``hyperseg_uav`` model and would reject 44 of the checkpoint's
tensors.  The sliding window below is copied verbatim from that exporter so the two
remain directly comparable: accumulate logits, divide by the cover count, argmax,
save as mode ``L``.

The window size is a parameter rather than a constant: the v2 training script is not
in the repository, so the resolution is calibrated by ``v2_val_replay.py`` against the
checkpoint's recorded ``val_miou`` and passed in with ``--size``.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from v2_model import load_v2  # noqa: E402

WORK = ROOT / 'runs/b3_resume_test2_20260930'
CHECKPOINT = ROOT / 'models/hyperseg_resume_best.pt'
INPUT = ROOT / 'dataset/low_altitude_2026/test_2/images'
OUTPUT = WORK / 'predictions'
EXPECTED_IMAGES = 1300
EXPECTED_SIDE = 1024
EXPECTED_CLASSES = 9
ZIP_NAME = 'hyperseg_b3_resume_test2_predictions.zip'


def png_side(path: Path):
    """Read width/height from the IHDR chunk without decoding the pixels."""
    with path.open('rb') as stream:
        header = stream.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise RuntimeError(f'Not a PNG file or missing IHDR: {path.name}')
    return struct.unpack('>II', header[16:24])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def sliding_logits(model, image, size, overlap=0.5):
    """Average logits over a sliding window; identical to tools/infer_hyperseg.py."""
    _, _, height, width = image.shape
    stride = max(1, int(size * (1.0 - overlap)))
    classes = model.model_config['classes']
    output = image.new_zeros((1, classes, height, width))
    weights = image.new_zeros((1, 1, height, width))
    tops = sorted(set(range(0, max(1, height - size + 1), stride)) | {max(0, height - size)})
    lefts = sorted(set(range(0, max(1, width - size + 1), stride)) | {max(0, width - size)})
    for top in tops:
        for left in lefts:
            bottom, right = min(height, top + size), min(width, left + size)
            crop = image[:, :, top:bottom, left:right]
            crop_shape = crop.shape[-2:]
            if crop_shape != (size, size):
                crop = torch.nn.functional.interpolate(
                    crop, (size, size), mode='bilinear', align_corners=False
                )
            crop_logits = model(crop)['logits']
            crop_logits = torch.nn.functional.interpolate(
                crop_logits, crop_shape, mode='bilinear', align_corners=False
            )
            output[:, :, top:bottom, left:right] += crop_logits
            weights[:, :, top:bottom, left:right] += 1
    return output / weights.clamp_min(1)


def preflight(size: int, overlap: float):
    images = sorted(INPUT.glob('*.png'))
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(f'Expected {EXPECTED_IMAGES} round-two images, found {len(images)}')
    names = [path.name for path in images]
    if len(set(names)) != len(names):
        raise RuntimeError('Duplicate input filenames')
    bad = [name for name, side in ((p.name, png_side(p)) for p in images)
           if side != (EXPECTED_SIDE, EXPECTED_SIDE)]
    if bad:
        raise RuntimeError(f'Unexpected input size for {len(bad)} images, e.g. {bad[:5]}')

    model, state, info = load_v2(CHECKPOINT, 'cpu')
    if info['model_config'].get('classes') != EXPECTED_CLASSES:
        raise RuntimeError(f'Expected a {EXPECTED_CLASSES}-class checkpoint')
    del model

    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': sha256(CHECKPOINT),
        'checkpoint_epoch': info['checkpoint_epoch'],
        'checkpoint_val_miou': info['checkpoint_val_miou'],
        'architecture': info['architecture'],
        'tensors_loaded': info['tensors_loaded'],
        'translated_legacy_encoder_keys': info['translated_legacy_encoder_keys'],
        'model_config': info['model_config'],
        'input': str(INPUT),
        'output': str(OUTPUT),
        'image_count': len(images),
        'image_side': EXPECTED_SIDE,
        'first_image': names[0],
        'last_image': names[-1],
        'size': size,
        'overlap': overlap,
        'tta': False,
        'precision': 'fp32',
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


@torch.no_grad()
def export(model, device, size: int, overlap: float):
    images = sorted(INPUT.glob('*.png'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, path in enumerate(images, start=1):
        array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
        logits = sliding_logits(model, image, size, overlap)
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        Image.fromarray(prediction, mode='L').save(OUTPUT / path.name)
        if index % 50 == 0 or index == len(images):
            rate = index / max(time.monotonic() - started, 1e-6)
            print(f'  {index}/{len(images)} ({rate:.1f} img/s)', flush=True)
    return time.monotonic() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, required=True,
                        help='window size; must match the calibrated training resolution')
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    os.chdir(ROOT)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'inference.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = preflight(args.size, args.overlap)
        if args.check_only:
            return
        if not config['cuda_available']:
            raise RuntimeError('No GPU allocated: enable GPU mode on server2 before launching')
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise RuntimeError('Predictions already exist; inspect them before starting another run')

        device = torch.device('cuda')
        model, _, _ = load_v2(CHECKPOINT, device)
        elapsed = export(model, device, args.size, args.overlap)
        del model
        torch.cuda.empty_cache()
        print(f'inference finished in {elapsed:.0f}s', flush=True)

        with (WORK / 'submission_check.log').open('w') as log:
            subprocess.run([sys.executable, 'tools/check_submission.py', str(OUTPUT), str(INPUT)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)

        submission = WORK / ZIP_NAME
        with zipfile.ZipFile(submission, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(OUTPUT.glob('*.png')):
                archive.write(path, arcname=path.name)
        summary = dict(config, status='complete', submission_check='passed',
                       submission=str(submission), submission_sha256=sha256(submission),
                       elapsed_seconds=elapsed)
        (WORK / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--check-only' not in sys.argv:
            (WORK / 'exit_code').write_text('1\n')
        raise
    else:
        if '--check-only' not in sys.argv:
            (WORK / 'exit_code').write_text('0\n')
