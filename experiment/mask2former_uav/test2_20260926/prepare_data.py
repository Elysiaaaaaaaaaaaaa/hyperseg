"""Verify and extract the competition round-two archive on server2."""
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'dataset/low_altitude_2026/test_2.zip'
DATA = ROOT / 'dataset/low_altitude_2026/test_2'
WORK = ROOT / 'runs/mask2former_uav/h3_test2_20260926'
EXPECTED_SHA256 = 'c8784dba28f19b86e7043bced92e42f98b346ae3f3ea13326f51ef6af295b65a'


def main():
    source = ARCHIVE if ARCHIVE.exists() else ARCHIVE.with_suffix('.zip.part')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('Archive SHA-256 differs from the local source')
    if source != ARCHIVE:
        source.rename(ARCHIVE)
    print('Archive SHA-256 verified:', digest.hexdigest(), flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    modes = Counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        names = [entry.filename for entry in entries]
        if len(names) != 1300 or len(set(names)) != 1300:
            raise RuntimeError('Expected 1300 unique images')
        for name in names:
            path = PurePosixPath(name)
            if path.parent != PurePosixPath('images') or path.suffix != '.png':
                raise RuntimeError(f'Unexpected archive member: {name}')
        expected_names = {PurePosixPath(name).name for name in names}
        image_dir = DATA / 'images'
        if image_dir.exists() and set(p.name for p in image_dir.iterdir()) - expected_names:
            raise RuntimeError('Destination contains files outside this archive')
        required = sum(entry.file_size for entry in entries if not (DATA / entry.filename).exists())
        if shutil.disk_usage(DATA).free < required + 512 * 1024 * 1024:
            raise RuntimeError('Insufficient free space for extraction and predictions')
        for index, entry in enumerate(entries, 1):
            target = DATA / entry.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.png.part')
            with archive.open(entry) as reader, temporary.open('wb') as writer:
                shutil.copyfileobj(reader, writer)
            with Image.open(temporary) as image:
                image.load()
                if image.format != 'PNG' or image.size != (1024, 1024):
                    raise RuntimeError(f'Invalid image format or dimensions: {entry.filename}')
                modes[image.mode] += 1
            temporary.replace(target)
            if index % 100 == 0:
                print(f'Validated {index}/{len(entries)} images', flush=True)
    manifest = {
        'archive': str(ARCHIVE), 'archive_sha256': EXPECTED_SHA256,
        'archive_bytes': ARCHIVE.stat().st_size,
        'input': str(DATA / 'images'), 'image_count': len(names),
        'image_size': [1024, 1024], 'image_modes': dict(modes),
        'image_names': sorted(expected_names), 'zip_crc_and_image_decode': 'passed',
        'has_ground_truth': False,
    }
    (WORK / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('READY: 1300 images validated; no labels, no mIoU evaluation.', flush=True)


if __name__ == '__main__':
    main()
