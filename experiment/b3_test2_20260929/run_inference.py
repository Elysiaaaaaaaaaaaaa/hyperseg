"""Run the existing HyperSeg-B3 exporter on the round-two test set, verify and package.

Protocol: FP32, 768x768 sliding window, 50% overlap, no TTA. 768 matches the
resolution used by tools/train_hyperseg.py for training and validation, so the
exported masks are produced at the same scale the checkpoint was selected on.
"""
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

import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'runs/b3_test2_20260929'
CHECKPOINT = ROOT / 'models/hyperseg_b3_best.pt'
INPUT = ROOT / 'dataset/low_altitude_2026/test_2/images'
OUTPUT = WORK / 'predictions'
SIZE = 768
OVERLAP = 0.5
EXPECTED_IMAGES = 1300
EXPECTED_SIDE = 1024
EXPECTED_CLASSES = 9
ZIP_NAME = 'hyperseg_b3_test2_predictions.zip'


def png_side(path):
    """Read width/height from the IHDR chunk without decoding the pixels."""
    with path.open('rb') as stream:
        header = stream.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise RuntimeError(f'Not a PNG file or missing IHDR: {path.name}')
    return struct.unpack('>II', header[16:24])


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preflight():
    from hyperseg_uav import HyperSegUAV, translate_legacy_keys  # noqa: F401

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
    if names[0] == names[-1]:
        raise RuntimeError('Input listing looks wrong')

    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    config = dict(state.get('model_config', {}))
    if config.get('classes') != EXPECTED_CLASSES:
        raise RuntimeError(f'Expected a {EXPECTED_CLASSES}-class checkpoint, got {config.get("classes")}')
    if 'model' not in state or 'head.weight' not in state['model']:
        raise RuntimeError('Checkpoint does not contain the expected HyperSeg-UAV weights')
    torch.manual_seed(0)
    model = HyperSegUAV(**dict(config, pretrained=False))
    state_dict, translated = translate_legacy_keys(state['model'])
    model.load_state_dict(state_dict)
    epoch, val_miou = state.get('epoch'), state.get('val_miou')
    del model, state

    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(CHECKPOINT),
        'checkpoint_sha256': sha256(CHECKPOINT),
        'checkpoint_epoch': epoch,
        'checkpoint_val_miou': val_miou,
        'translated_legacy_encoder_keys': translated,
        'model_config': config,
        'input': str(INPUT),
        'output': str(OUTPUT),
        'image_count': len(images),
        'image_side': EXPECTED_SIDE,
        'first_image': names[0],
        'last_image': names[-1],
        'size': SIZE,
        'overlap': OVERLAP,
        'tta': False,
        'precision': 'fp32',
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (WORK / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    with (WORK / 'inference.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = preflight()
        if args.check_only:
            return
        if not config['cuda_available']:
            raise RuntimeError('No GPU allocated: enable GPU mode on server3 before launching')
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise RuntimeError('Predictions already exist; inspect them before starting another run')
        started = time.monotonic()
        command = [sys.executable, '-u', 'tools/infer_hyperseg.py',
                   '--checkpoint', str(CHECKPOINT), '--input', str(INPUT),
                   '--output', str(OUTPUT), '--size', str(SIZE),
                   '--overlap', str(OVERLAP)]
        print('Starting:', ' '.join(command), flush=True)
        with (WORK / 'infer.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    result = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'Progress: {len(list(OUTPUT.glob("*.png")))}/{EXPECTED_IMAGES}', flush=True)
        if result:
            raise RuntimeError(f'Inference exited with {result}; see {WORK / "infer.log"}')
        with (WORK / 'submission_check.log').open('w') as log:
            subprocess.run([sys.executable, 'tools/check_submission.py', str(OUTPUT), str(INPUT)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        submission = WORK / ZIP_NAME
        with zipfile.ZipFile(submission, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(OUTPUT.glob('*.png')):
                archive.write(path, arcname=path.name)
        summary = dict(config, status='complete', submission_check='passed',
                       submission=str(submission), submission_sha256=sha256(submission),
                       elapsed_seconds=time.monotonic() - started)
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
