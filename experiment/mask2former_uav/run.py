"""Prepare, check, train, evaluate, and export the UAV MMSeg experiments."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = ('mask2former', 'segformer')
SPLITS = ('train', 'val', 'test')
H3_SCRIPT = HERE / 'train_h3_hyperseg.py'


def config_path(model: str, submission: bool = False) -> Path:
    suffix = '_submission' if submission else ''
    return HERE / f'{model}_mit_b3_512{suffix}.py'


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
              if line.strip()]
    if not values:
        raise ValueError(f'Empty split: {path}')
    with_suffix = [value for value in values if Path(value).suffix]
    if with_suffix:
        raise ValueError(
            f'Split entries must be stems without .png: {with_suffix[:3]} in {path}')
    if len(values) != len(set(values)):
        raise ValueError(f'Duplicate IDs in {path}')
    return values


def check_splits(split_dir: Path) -> dict[str, list[str]]:
    result = {name: read_ids(split_dir / f'{name}.txt') for name in SPLITS}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = set(result[left]) & set(result[right])
            if overlap:
                raise ValueError(
                    f'{left}/{right} split overlap: {sorted(overlap)[:5]}')
    return result


def check_dataset(data_root: Path, split_dir: Path, scan_masks: bool) -> None:
    splits = check_splits(split_dir)
    image_dir = data_root / 'train' / 'images'
    mask_dir = data_root / 'train' / 'masks'
    missing_images = []
    missing_masks = []
    for sample_id in set().union(*map(set, splits.values())):
        if not (image_dir / f'{sample_id}.png').is_file():
            missing_images.append(sample_id)
        if not (mask_dir / f'{sample_id}.png').is_file():
            missing_masks.append(sample_id)
    if missing_images or missing_masks:
        raise FileNotFoundError(
            f'Missing images={missing_images[:5]} ({len(missing_images)} total), '
            f'masks={missing_masks[:5]} ({len(missing_masks)} total)')

    if scan_masks:
        from PIL import Image

        invalid = []
        for sample_id in sorted(set().union(*map(set, splits.values()))):
            with Image.open(mask_dir / f'{sample_id}.png') as mask:
                labels = set(mask.convert('L').getdata())
            if not labels <= set(range(9)):
                invalid.append((sample_id, sorted(labels - set(range(9)))))
        if invalid:
            raise ValueError(f'Masks contain labels outside 0..8: {invalid[:5]}')
    counts = ', '.join(f'{name}={len(ids)}' for name, ids in splits.items())
    print(f'Dataset preflight OK: {counts}', flush=True)


def runtime_env(args) -> dict[str, str]:
    env = os.environ.copy()
    env['HYPERSEG_PROJECT_ROOT'] = str(ROOT.resolve())
    env['HYPERSEG_DATA_ROOT'] = str(args.data_root.resolve())
    env['HYPERSEG_SPLIT_DIR'] = str(args.split_dir.resolve())
    if getattr(args, 'backbone_checkpoint', None):
        checkpoint = args.backbone_checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Missing backbone checkpoint: {checkpoint}')
        env['HYPERSEG_MIT_B3_CHECKPOINT'] = str(checkpoint)
    if getattr(args, 'input', None):
        env['HYPERSEG_TEST_IMAGE_DIR'] = str(args.input.resolve())
    current = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(args.mmseg_root.resolve()) + (os.pathsep + current if current else '')
    return env


def check_runtime(python: Path, mmseg_root: Path, env: dict[str, str]) -> None:
    train_script = mmseg_root / 'tools' / 'train.py'
    test_script = mmseg_root / 'tools' / 'test.py'
    if not train_script.is_file() or not test_script.is_file():
        raise FileNotFoundError(f'Invalid MMSegmentation checkout: {mmseg_root}')
    probe = (
        'import mmcv, mmengine, mmdet, mmseg, torch; '
        'print("runtime:", "torch="+torch.__version__, "mmcv="+mmcv.__version__, '
        '"mmengine="+mmengine.__version__, "mmdet="+mmdet.__version__, '
        '"mmseg="+mmseg.__version__)')
    subprocess.run([str(python), '-c', probe], check=True, env=env)


def cfg_options(args) -> list[str]:
    return [
        f'train_dataloader.batch_size={args.batch_size}',
        f'train_dataloader.num_workers={args.num_workers}',
        f'val_dataloader.num_workers={args.num_workers}',
        f'test_dataloader.num_workers={args.num_workers}',
        f'optim_wrapper.accumulative_counts={args.accumulative_counts}',
        f'randomness.seed={args.seed}',
        f'train_cfg.max_iters={args.max_iters}',
        f'train_cfg.val_interval={args.val_interval}',
        f'param_scheduler.0.end={args.max_iters}',
        f'default_hooks.checkpoint.interval={args.val_interval}',
    ]


def show_or_run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print('command:', shlex.join(command), flush=True)
    for name in ('HYPERSEG_PROJECT_ROOT', 'HYPERSEG_DATA_ROOT', 'HYPERSEG_SPLIT_DIR',
                 'HYPERSEG_MIT_B3_CHECKPOINT', 'HYPERSEG_TEST_IMAGE_DIR'):
        if name in env:
            print(f'{name}={env[name]}', flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT, env=env)


def prepare_splits(args) -> None:
    members = {f'splits/{name}.txt': name for name in SPLITS}
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        name: args.output_dir / f'{name}.txt' for name in SPLITS}
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f'{existing[0]} exists; pass --force to replace all splits')
    with zipfile.ZipFile(args.archive) as archive:
        missing = set(members) - set(archive.namelist())
        if missing:
            raise ValueError(f'Archive is missing: {sorted(missing)}')
        for member, name in members.items():
            lines = archive.read(member).decode('utf-8').splitlines()
            normalized = [Path(line.strip()).stem for line in lines if line.strip()]
            destinations[name].write_text(
                '\n'.join(normalized) + '\n', encoding='utf-8')
    splits = check_splits(args.output_dir)
    print('Prepared splits: ' + ', '.join(
        f'{name}={len(values)}' for name, values in splits.items()))


def train(args) -> None:
    if not args.dry_run:
        check_dataset(args.data_root, args.split_dir, scan_masks=False)
    env = runtime_env(args)
    if not args.dry_run:
        check_runtime(args.python, args.mmseg_root, env)
    work_dir = args.work_dir or (
        ROOT / 'runs' / 'mask2former_uav' / f'{args.model}_mit_b3_512')
    command = [
        str(args.python), '-u', str(args.mmseg_root / 'tools' / 'train.py'),
        str(config_path(args.model)), '--work-dir', str(work_dir),
        '--cfg-options', *cfg_options(args),
    ]
    if args.amp:
        command.append('--amp')
    if args.resume:
        command.append('--resume')
    show_or_run(command, env, args.dry_run)


def train_h3(args) -> None:
    """Launch the standalone Swin-L + HyperSeg experiment entry point."""
    env = runtime_env(args)
    command = [
        str(args.python), '-u', str(H3_SCRIPT),
        '--data-root', str(args.data_root),
        '--split-dir', str(args.split_dir),
        '--work-dir', str(args.work_dir),
        '--batch-size', str(args.batch_size),
        '--accumulative-counts', str(args.accumulative_counts),
        '--max-iters', str(args.max_iters),
        '--val-interval', str(args.val_interval),
        '--num-workers', str(args.num_workers),
        '--seed', str(args.seed),
        '--size', str(args.size),
    ]
    if args.backbone_checkpoint:
        command.extend(['--backbone-checkpoint', str(args.backbone_checkpoint)])
    if args.no_pretrained:
        command.append('--no-pretrained')
    if args.with_checkpointing:
        command.append('--with-checkpointing')
    if args.resume:
        command.extend(['--resume', str(args.resume)])
    show_or_run(command, env, args.dry_run)


def evaluate(args) -> None:
    if not args.dry_run:
        check_dataset(args.data_root, args.split_dir, scan_masks=False)
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    env = runtime_env(args)
    if not args.dry_run:
        check_runtime(args.python, args.mmseg_root, env)
    command = [
        str(args.python), '-u', str(args.mmseg_root / 'tools' / 'test.py'),
        str(config_path(args.model)), str(args.checkpoint),
        '--work-dir', str(args.work_dir), '--cfg-options',
        f'test_dataloader.num_workers={args.num_workers}',
        'model.backbone.init_cfg=None',
    ]
    show_or_run(command, env, args.dry_run)


def export(args) -> None:
    if not args.dry_run:
        if not args.input.is_dir():
            raise FileNotFoundError(args.input)
        if not list(args.input.glob('*.png')):
            raise ValueError(f'No PNG images found in {args.input}')
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
    env = runtime_env(args)
    if not args.dry_run:
        check_runtime(args.python, args.mmseg_root, env)
    command = [
        str(args.python), '-u', str(args.mmseg_root / 'tools' / 'test.py'),
        str(config_path(args.model, submission=True)), str(args.checkpoint),
        '--work-dir', str(args.work_dir), '--out', str(args.output),
        '--cfg-options', f'test_dataloader.num_workers={args.num_workers}',
        'model.backbone.init_cfg=None',
    ]
    show_or_run(command, env, args.dry_run)
    if not args.dry_run:
        subprocess.run(
            [str(args.python), str(ROOT / 'tools' / 'check_submission.py'),
             str(args.output), str(args.input)],
            check=True, cwd=ROOT, env=env)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--mmseg-root', type=Path, default=ROOT.parent / 'mmsegmentation')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--split-dir', type=Path, default=ROOT / 'runs' / 'splits')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare = subparsers.add_parser('prepare-splits')
    prepare.add_argument('--archive', type=Path, default=ROOT / 'splits.zip')
    prepare.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / 'splits')
    prepare.add_argument('--force', action='store_true')
    prepare.set_defaults(function=prepare_splits)

    check = subparsers.add_parser('check')
    check.add_argument('--data-root', type=Path, default=ROOT / 'dataset')
    check.add_argument('--split-dir', type=Path, default=ROOT / 'runs' / 'splits')
    check.add_argument('--scan-masks', action='store_true')
    check.set_defaults(
        function=lambda args: check_dataset(
            args.data_root, args.split_dir, args.scan_masks))

    train_parser = subparsers.add_parser('train')
    add_runtime_arguments(train_parser)
    train_parser.add_argument('--model', choices=MODELS, default='mask2former')
    train_parser.add_argument(
        '--work-dir', type=Path,
        help='Defaults to runs/mask2former_uav/<model>_mit_b3_512')
    train_parser.add_argument('--backbone-checkpoint', type=Path)
    train_parser.add_argument('--batch-size', type=int, default=2)
    train_parser.add_argument('--accumulative-counts', type=int, default=1)
    train_parser.add_argument('--max-iters', type=int, default=160000)
    train_parser.add_argument('--val-interval', type=int, default=2800)
    train_parser.add_argument('--seed', type=int, default=3407)
    train_parser.add_argument(
        '--amp', action=argparse.BooleanOptionalAction, default=False)
    train_parser.add_argument('--resume', action='store_true')
    train_parser.set_defaults(function=train)

    h3_parser = subparsers.add_parser(
        'train-h3', help='train Swin-L with the repository HyperSeg decoder')
    add_runtime_arguments(h3_parser)
    h3_parser.add_argument('--work-dir', type=Path, default=ROOT / 'runs' / 'mask2former_uav' / 'h3_swin_l_hyperseg_512_seed3407')
    h3_parser.add_argument('--backbone-checkpoint', type=Path)
    h3_parser.add_argument('--batch-size', type=int, default=2)
    h3_parser.add_argument('--accumulative-counts', type=int, default=1)
    h3_parser.add_argument('--max-iters', type=int, default=160000)
    h3_parser.add_argument('--val-interval', type=int, default=2800)
    h3_parser.add_argument('--size', type=int, default=512)
    h3_parser.add_argument('--seed', type=int, default=3407)
    h3_parser.add_argument('--no-pretrained', action='store_true')
    h3_parser.add_argument('--with-checkpointing', action='store_true')
    h3_parser.add_argument('--resume', type=Path)
    h3_parser.set_defaults(function=train_h3)

    eval_parser = subparsers.add_parser('eval')
    add_runtime_arguments(eval_parser)
    eval_parser.add_argument('--model', choices=MODELS, default='mask2former')
    eval_parser.add_argument('--checkpoint', type=Path, required=True)
    eval_parser.add_argument(
        '--work-dir', type=Path,
        default=ROOT / 'runs' / 'mask2former_uav' / 'evaluation')
    eval_parser.set_defaults(function=evaluate)

    export_parser = subparsers.add_parser('export')
    add_runtime_arguments(export_parser)
    export_parser.add_argument('--model', choices=MODELS, default='mask2former')
    export_parser.add_argument('--checkpoint', type=Path, required=True)
    export_parser.add_argument(
        '--input', type=Path,
        default=ROOT / 'dataset' / 'low_altitude_2026' / 'images')
    export_parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'runs' / 'mask2former_uav' / 'predictions')
    export_parser.add_argument(
        '--work-dir', type=Path,
        default=ROOT / 'runs' / 'mask2former_uav' / 'submission_test')
    export_parser.set_defaults(function=export)

    args = parser.parse_args()
    for name in ('num_workers', 'batch_size', 'accumulative_counts',
                 'max_iters', 'val_interval'):
        if hasattr(args, name) and getattr(args, name) < (0 if name == 'num_workers' else 1):
            parser.error(f'--{name.replace("_", "-")} has an invalid value')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    arguments.function(arguments)
