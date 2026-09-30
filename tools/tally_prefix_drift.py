"""Tally which attribute drifted for each mismatching member of a paused prefix.

The comparison tool prints one line per mismatch; 11k lines are hard to read. This reruns
it and reports the drift per field, which is what actually decides the verdict: ``atime``
drift can in principle be reset with ``os.utime`` — except that resetting it moves
``ctime``, which the same pax records also pin down and which no userspace tool can set.
So a prefix whose mismatches are atime-driven is still unusable.

Writes the raw lines to the path given as the second argument for the record.
"""

import collections
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

entries = sys.argv[1]
out_path = Path(sys.argv[2])

result = subprocess.run(
    [PYTHON, str(PROJECT / "tools" / "compare_prefix_attrs.py"), entries],
    capture_output=True, text=True, encoding="utf-8", errors="replace")
out_path.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr,
                    encoding="utf-8")

fields = collections.Counter()
kinds = collections.Counter()
patterns = {
    "type": re.compile(r"type "),
    "mode": re.compile(r"mode 0o"),
    "owner": re.compile(r"owner "),
    "size": re.compile(r"size \d+ ->"),
    "mtime": re.compile(r"mtime \d"),
    "atime": re.compile(r"atime \d"),
    "ctime": re.compile(r"ctime \d"),
}
for line in result.stdout.splitlines():
    if "\t" not in line:
        continue
    name, _, problems = line.partition("\t")
    seen = []
    for field, pattern in patterns.items():
        if pattern.search(problems):
            fields[field] += 1
            seen.append(field)
    kinds["+".join(seen)] += 1

print(f"lines: {sum(kinds.values())}")
print("\ndrift by field:")
for field, count in fields.most_common():
    print(f"  {field:6} {count}")
print("\ndrift combinations:")
for combo, count in kinds.most_common(12):
    print(f"  {count:6}  {combo}")
print("\nfirst 3 examples of each combination:")
shown = collections.Counter()
for line in result.stdout.splitlines():
    if "\t" not in line:
        continue
    name, _, problems = line.partition("\t")
    seen = "+".join(f for f, p in patterns.items() if p.search(problems))
    if shown[seen] < 3:
        shown[seen] += 1
        print(f"  [{seen}] {name}\n      {problems}")
print(f"\nraw comparison output: {out_path}")
