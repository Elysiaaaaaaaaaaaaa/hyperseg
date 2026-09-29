"""Verify the system-disk competition copy against server2's data disk."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE = Path('/root/autodl-tmp')
PROJECT = Path('/root/hyperseg')
SOURCE_DATA = SOURCE / 'data/low_altitude_2026'
DATA = PROJECT / 'dataset/low_altitude_2026'
SOURCE_CHECKPOINT = SOURCE / 'work_dirs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt'
CHECKPOINT = PROJECT / 'runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt'
WORK = PROJECT / 'runs/mask2former_uav/h3_test2_20260926'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def names(path: Path) -> set[str]:
    return {item.name for item in path.glob('*.png')}


def checksum_tree(relative: str) -> None:
    source = SOURCE_DATA / relative
    target = DATA / relative
    result = subprocess.run(
        ['rsync', '-a', '--checksum', '--dry-run', '--delete',
         '--itemize-changes', f'{source}/', f'{target}/'],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f'Copy mismatch in {relative}: {result.stdout[:1000]}')
    print(f'Checksum tree OK: {relative}', flush=True)


def main() -> None:
    counts = {'train/images': 6996, 'train/masks': 6996,
              'images': 500, 'test_2/images': 1300}
    for relative, expected in counts.items():
        actual = len(names(DATA / relative))
        if actual != expected:
            raise RuntimeError(f'{relative}: expected {expected} PNGs, found {actual}')
    if names(DATA / 'train/images') != names(DATA / 'train/masks'):
        raise RuntimeError('Labeled image and mask filenames differ')
    splits = {
        split: set((PROJECT / 'runs/splits' / f'{split}.txt').read_text().splitlines())
        for split in ('train', 'val', 'test')
    }
    if [len(splits[split]) for split in ('train', 'val', 'test')] != [5598, 699, 699]:
        raise RuntimeError('Fixed split counts differ')
    if len(set.union(*splits.values())) != 6996:
        raise RuntimeError('Fixed splits overlap or omit labeled samples')
    if {f'{stem}.png' for stem in set.union(*splits.values())} != names(DATA / 'train/images'):
        raise RuntimeError('Fixed splits do not cover labeled images')
    for relative in counts:
        checksum_tree(relative)
    for relative in ('Label.txt', 'stratified_split_90_10.json'):
        source, target = SOURCE_DATA / relative, DATA / relative
        if source.is_file() and sha256(source) != sha256(target):
            raise RuntimeError(f'Metadata mismatch: {relative}')
    for split in ('train', 'val', 'test'):
        relative = f'runs/splits/{split}.txt'
        if sha256(SOURCE / 'hyperseg' / relative) != sha256(PROJECT / relative):
            raise RuntimeError(f'Split mismatch: {split}')
    source_hash = sha256(SOURCE_CHECKPOINT)
    if source_hash != sha256(CHECKPOINT):
        raise RuntimeError('H3 checkpoint differs from source')
    print(f'Checkpoint SHA-256 OK: {source_hash}', flush=True)

    mmseg_check = subprocess.run(
        ['rsync', '-a', '--checksum', '--dry-run', '--delete', '--itemize-changes',
         f'{SOURCE / "mmsegmentation"}/', '/root/mmsegmentation/'],
        check=True, capture_output=True, text=True,
    )
    if mmseg_check.stdout.strip():
        raise RuntimeError('MMSegmentation checkout differs from source')

    source_manifest = json.loads((SOURCE / 'work_dirs/mask2former_uav/h3_test2_20260926/data_manifest.json').read_text())
    if set(source_manifest['image_names']) != names(DATA / 'test_2/images'):
        raise RuntimeError('Round-two filenames differ from validated source manifest')
    manifest = dict(source_manifest)
    manifest['archive'] = None  # The duplicate source ZIP is intentionally omitted.
    manifest['input'] = str(DATA / 'test_2/images')
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    python = PROJECT / '.venv-mask2former/bin/python'
    env = os.environ.copy()
    env['PYTHONPATH'] = f'/root/mmsegmentation:{PROJECT}'
    probe = ('import sys, torch, mmcv, mmengine, mmdet, mmseg; '
             'from mmseg.models.backbones import SwinTransformer; '
             'print("prefix", sys.prefix); '
             'print("versions", torch.__version__, mmcv.__version__, '
             'mmengine.__version__, mmdet.__version__, mmseg.__version__)')
    subprocess.run([str(python), '-c', probe], check=True, env=env, cwd=PROJECT)
    subprocess.run([str(python),
                    str(PROJECT / 'experiment/mask2former_uav/test2_20260926/run_inference.py'),
                    '--check-only'], check=True, env=env, cwd=PROJECT)

    report = {
        'status': 'passed', 'project': str(PROJECT), 'data': str(DATA),
        'image_counts': counts, 'split_counts': {key: len(value) for key, value in splits.items()},
        'checkpoint_sha256': source_hash,
        'checkpoint_bytes': CHECKPOINT.stat().st_size,
        'runtime_check': 'passed', 'h3_preflight': 'passed',
    }
    target = PROJECT / 'runs/system_image_20260926/verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(f'System-disk migration verified: {target}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'VERIFICATION FAILED: {error}', file=sys.stderr, flush=True)
        raise
