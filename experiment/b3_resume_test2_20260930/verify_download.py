"""Verify the downloaded round-two prediction ZIP against the local test_2 archive."""
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / 'experiment/b3_resume_test2_20260930/hyperseg_b3_resume_test2_predictions.zip'
SOURCE = ROOT / 'test_2.zip'
EXPECTED_SHA = 'ff0a2fb0e0580bafb1b7f41f82d29861ad42bf181689f7efff43df3e84b0ef8a'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    report = {}
    report['submission'] = str(SUBMISSION)
    report['submission_bytes'] = SUBMISSION.stat().st_size
    report['sha256'] = sha256(SUBMISSION)
    report['sha256_matches_server'] = report['sha256'] == EXPECTED_SHA

    with zipfile.ZipFile(SUBMISSION) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        report['entries'] = len(names)
        report['crc_ok'] = bad is None
        report['duplicate_entries'] = len(names) - len(set(names))
        report['nested_paths'] = sum('/' in name for name in names)
        histogram = collections.Counter()
        sizes, modes = set(), set()
        for name in names:
            with archive.open(name) as stream:
                image = Image.open(stream)
                sizes.add(image.size)
                modes.add(image.mode)
                histogram.update(image.getdata())
        report['modes'] = sorted(modes)
        report['sizes'] = sorted(str(size) for size in sizes)
        report['class_ids'] = sorted(histogram)
        report['pixel_counts'] = {str(key): histogram[key] for key in sorted(histogram)}
        report['share'] = {str(key): round(histogram[key] / sum(histogram.values()), 6)
                           for key in sorted(histogram)}
    with zipfile.ZipFile(SOURCE) as source:
        source_names = [name.split('/')[-1] for name in source.namelist() if name.endswith('.png')]
    report['source_images'] = len(source_names)
    report['names_match_source'] = set(source_names) == set(names)
    report['missing_from_prediction'] = sorted(set(source_names) - set(names))[:5]
    report['extra_in_prediction'] = sorted(set(names) - set(source_names))[:5]

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'experiment/b3_resume_test2_20260930/logs/download_verification.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
