"""Check the round-two prediction ZIP against section 13 of the competition manual.

Manual requirements (initial round, reused verbatim for round two):
  * prediction file names identical to the test-set file names;
  * pixel values strictly the class IDs 0..8;
  * single-channel PNG only - RGB, palette/indexed PNG and other non-standard
    formats are rejected;
  * resolution identical to the source image (all 1024x1024), no rescaling;
  * everything packed into one zip file.

The checks below read the raw PNG chunks instead of trusting Pillow's mode,
so a palette (PLTE) or transparency (tRNS) chunk cannot slip through.
"""
import collections
import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / 'experiment/b3_test2_20260929/hyperseg_b3_test2_predictions.zip'
SOURCE = ROOT / 'test_2.zip'
EXPECTED_CLASSES = set(range(9))
EXPECTED_SIDE = (1024, 1024)


def png_chunks(data):
    """Yield (type, payload) for every chunk in a PNG byte string."""
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('missing PNG signature')
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8].decode('ascii', 'replace')
        payload = data[offset + 8:offset + 8 + length]
        yield chunk_type, payload
        offset += 12 + length
        if chunk_type == 'IEND':
            break


def main():
    report = {'submission': str(SUBMISSION), 'source': str(SOURCE)}
    report['bytes'] = SUBMISSION.stat().st_size
    report['sha256'] = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()

    with zipfile.ZipFile(SOURCE) as archive:
        source_names = [Path(name).name for name in archive.namelist() if name.endswith('.png')]

    failures = []
    counts = collections.Counter()
    histogram = collections.Counter()
    with zipfile.ZipFile(SUBMISSION) as archive:
        report['crc_ok'] = archive.testzip() is None
        names = archive.namelist()
        report['entries'] = len(names)
        report['non_png_entries'] = [n for n in names if not n.lower().endswith('.png')]
        report['nested_or_dir_entries'] = [n for n in names if '/' in n or '\\' in n or n.endswith('/')]
        report['duplicate_entries'] = sorted({n for n in names if names.count(n) > 1})
        report['name_set_matches_test_set_2'] = set(names) == set(source_names)
        report['missing_names'] = sorted(set(source_names) - set(names))[:5]
        report['extra_names'] = sorted(set(names) - set(source_names))[:5]

        for name in names:
            counts['files'] += 1
            data = archive.read(name)
            chunk_types = []
            ihdr = None
            for chunk_type, payload in png_chunks(data):
                chunk_types.append(chunk_type)
                if chunk_type == 'IHDR':
                    ihdr = struct.unpack('>IIBBBBB', payload)
            width, height, depth, color_type, compression, filter_type, interlace = ihdr
            if color_type == 0:
                counts['color_type_gray'] += 1
            elif color_type == 3:
                counts['color_type_palette'] += 1
                failures.append(f'{name}: palette/indexed PNG (color type 3)')
            else:
                counts[f'color_type_{color_type}'] += 1
                failures.append(f'{name}: color type {color_type} (not single-channel gray)')
            if depth != 8:
                counts[f'bit_depth_{depth}'] += 1
                failures.append(f'{name}: bit depth {depth}, expected 8')
            if 'PLTE' in chunk_types:
                counts['with_plte_chunk'] += 1
                failures.append(f'{name}: contains PLTE chunk')
            if 'tRNS' in chunk_types:
                counts['with_trns_chunk'] += 1
                failures.append(f'{name}: contains tRNS chunk')
            if interlace != 0:
                failures.append(f'{name}: interlaced PNG')
            if (width, height) != EXPECTED_SIDE:
                failures.append(f'{name}: size {width}x{height}, expected 1024x1024')

            with Image.open(io.BytesIO(data)) as image:
                if image.mode != 'L':
                    failures.append(f'{name}: Pillow mode {image.mode}, expected L')
                if image.size != EXPECTED_SIDE:
                    failures.append(f'{name}: Pillow size {image.size}')
                pixels = image.tobytes()
                values = set(pixels)
                if not values <= EXPECTED_CLASSES:
                    failures.append(f'{name}: class ids outside 0..8 -> {sorted(values)[:8]}')
                if len(pixels) != EXPECTED_SIDE[0] * EXPECTED_SIDE[1]:
                    failures.append(f'{name}: {len(pixels)} pixels, expected 1048576')
                histogram.update(pixels)

    report['counts'] = dict(counts)
    report['failures'] = failures
    report['failure_count'] = len(failures)
    report['class_ids'] = sorted(histogram)
    report['pixel_share'] = {str(k): round(histogram[k] / sum(histogram.values()), 6)
                             for k in sorted(histogram)}
    verdict = (
        'PASS: zip matches the manual section 13 requirements'
        if not failures
        and report['crc_ok']
        and report['name_set_matches_test_set_2']
        and report['entries'] == len(source_names)
        and report['entries'] == 1300
        else 'FAIL'
    )
    report['verdict'] = verdict
    out = ROOT / 'experiment/b3_test2_20260929/logs/manual_compliance_20260929.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'pixel_share'},
                     indent=2, ensure_ascii=False))
    print('pixel_share =', json.dumps(report['pixel_share'], ensure_ascii=False))
    sys.exit(0 if verdict.startswith('PASS') else 1)


if __name__ == '__main__':
    main()
