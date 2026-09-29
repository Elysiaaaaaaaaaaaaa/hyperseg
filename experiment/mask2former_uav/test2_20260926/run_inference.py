"""Run the existing H3 exporter, check every mask, and package round-two results."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'runs/mask2former_uav/h3_test2_20260926'
CHECKPOINT = ROOT / 'runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt'
INPUT = ROOT / 'dataset/low_altitude_2026/test_2/images'
OUTPUT = WORK / 'predictions'


def preflight():
    from hyperseg_uav.swin_l_model import SwinLHyperSeg  # noqa: F401
    from mmseg.models.backbones import SwinTransformer  # noqa: F401

    manifest = json.loads((WORK / 'data_manifest.json').read_text())
    if set(p.name for p in INPUT.glob('*.png')) != set(manifest['image_names']):
        raise RuntimeError('Input filenames no longer match the validated archive')
    # The training checkpoint includes optimizer tensors (~2.3 GiB). Metadata
    # checks must also work in AutoDL CPU mode with its small memory quota.
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    if state.get('step') != 137200 or state.get('experiment') != 'H3_swin_l_hyperseg':
        raise RuntimeError('Unexpected checkpoint identity')
    if state['model_config']['classes'] != 9:
        raise RuntimeError('Expected the original 9-channel UAV checkpoint')
    effective_config = dict(state['model_config'])
    effective_config['pretrained_checkpoint'] = None
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(CHECKPOINT), 'step': state['step'],
        'best_val_mIoU': state['best_mIoU'], 'model_config': effective_config,
        'input': str(INPUT), 'output': str(OUTPUT), 'image_count': manifest['image_count'],
        'archive_sha256': manifest['archive_sha256'],
        'size': 512, 'overlap': 0.5, 'tta': False, 'precision': 'fp32',
        'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    del state
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
            raise RuntimeError('No GPU allocated: enable GPU mode on server2 before launching')
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise RuntimeError('Predictions already exist; inspect them before starting another run')
        started = time.monotonic()
        command = [sys.executable, '-u', 'experiment/mask2former_uav/infer_h3_hyperseg.py',
                   '--checkpoint', str(CHECKPOINT), '--input', str(INPUT),
                   '--output', str(OUTPUT), '--size', '512', '--overlap', '0.5']
        print('Starting:', ' '.join(command), flush=True)
        with (WORK / 'infer.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    result = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'Progress: {len(list(OUTPUT.glob("*.png")))}/1300', flush=True)
        if result:
            raise RuntimeError(f'Inference exited with {result}; see {WORK / "infer.log"}')
        with (WORK / 'submission_check.log').open('w') as log:
            subprocess.run([sys.executable, 'tools/check_submission.py', str(OUTPUT), str(INPUT)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        submission = WORK / 'h3_test2_predictions.zip'
        with zipfile.ZipFile(submission, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(OUTPUT.glob('*.png')):
                archive.write(path, arcname=path.name)
        digest = hashlib.sha256()
        with submission.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        summary = dict(config, status='complete', submission_check='passed',
                       submission=str(submission), submission_sha256=digest.hexdigest(),
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
