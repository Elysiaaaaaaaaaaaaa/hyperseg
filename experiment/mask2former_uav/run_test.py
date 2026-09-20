"""Export H1 Mask2Former predictions for the unlabelled competition test set.

The wrapper locates the test images and the completed H1 checkpoint on either
the local checkout or the server layout, then delegates inference and output
validation to ``run.py export``.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_NAME = 'h1_mit_b3_160k_seed3407_fp32'


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _executable_path(path: Path) -> Path:
    """Make an executable absolute without resolving a virtualenv symlink."""
    return path.expanduser().absolute()


def resolve_input(explicit: Optional[Path]) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_path = os.environ.get('HYPERSEG_TEST_IMAGE_DIR')
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        ROOT / 'dataset' / 'low_altitude_2026' / 'images',
        ROOT.parent / 'data' / 'low_altitude_2026' / 'images',
        ROOT.parent / 'dataset' / 'low_altitude_2026' / 'images',
        ROOT / 'dataset' / 'test_1' / 'images',
    ])

    checked = []
    for candidate in candidates:
        path = _resolved(candidate)
        if path in checked:
            continue
        checked.append(path)
        if path.is_dir() and any(path.glob('*.png')):
            return path
    attempted = '\n  '.join(str(path) for path in checked)
    raise FileNotFoundError(
        'Could not find a test image directory containing PNG files. '
        f'Pass --input explicitly. Checked:\n  {attempted}')


def _checkpoint_iteration(path: Path) -> int:
    match = re.search(r'_iter_(\d+)\.pth$', path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(explicit: Optional[Path], dry_run: bool) -> Path:
    if explicit is not None:
        path = _resolved(explicit)
        if not dry_run and not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {path}')
        return path

    env_path = os.environ.get('HYPERSEG_MASK2FORMER_CHECKPOINT')
    if env_path:
        path = _resolved(Path(env_path))
        if path.is_file():
            return path

    work_dirs = [
        ROOT.parent / 'work_dirs' / 'mask2former_uav' / RUN_NAME,
        ROOT / 'runs' / 'mask2former_uav' / RUN_NAME,
    ]
    matches = [
        checkpoint
        for work_dir in work_dirs
        for checkpoint in work_dir.glob('best_mIoU_iter_*.pth')
        if checkpoint.is_file()
    ]
    if matches:
        return max(matches, key=_checkpoint_iteration).resolve()

    attempted = '\n  '.join(str(path.resolve()) for path in work_dirs)
    raise FileNotFoundError(
        'Could not find an H1 best checkpoint. Pass --checkpoint explicitly '
        'or set HYPERSEG_MASK2FORMER_CHECKPOINT. Searched:\n  '
        f'{attempted}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input', type=Path,
        help='Test image directory; auto-detected when omitted.')
    parser.add_argument(
        '--checkpoint', type=Path,
        help='H1 checkpoint; auto-detected from the completed work directory.')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'runs' / 'mask2former_uav' / RUN_NAME / 'test_predictions')
    parser.add_argument(
        '--work-dir', type=Path,
        default=ROOT / 'runs' / 'mask2former_uav' / RUN_NAME / 'test_work_dir')
    parser.add_argument('--mmseg-root', type=Path,
                        default=ROOT.parent / 'mmsegmentation')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    return args


def main() -> None:
    args = parse_args()
    input_dir = resolve_input(args.input)
    checkpoint = resolve_checkpoint(args.checkpoint, args.dry_run)
    image_count = sum(1 for _ in input_dir.glob('*.png'))

    command = [
        str(_executable_path(args.python)),
        str(HERE / 'run.py'),
        'export',
        '--model', 'mask2former',
        '--python', str(_executable_path(args.python)),
        '--mmseg-root', str(_resolved(args.mmseg_root)),
        '--checkpoint', str(checkpoint),
        '--input', str(input_dir),
        '--output', str(_resolved(args.output)),
        '--work-dir', str(_resolved(args.work_dir)),
        '--num-workers', str(args.num_workers),
    ]
    if args.dry_run:
        command.append('--dry-run')

    print(f'test images: {image_count} from {input_dir}', flush=True)
    print(f'checkpoint: {checkpoint}', flush=True)
    print(f'predictions: {_resolved(args.output)}', flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
