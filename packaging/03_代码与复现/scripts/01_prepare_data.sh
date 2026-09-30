#!/usr/bin/env bash
# Put the official data into the layout tools/train_hyperseg.py expects.
#
#   tools/train_hyperseg.py reads   <data>/train/train/images   and  .../masks
#   the official data is laid out   low_altitude_2026/train/images  and  .../masks
#
# so one directory level has to be bridged. Symlinks keep this free of copies.
#
# The work dir lives inside the project root (03_代码与复现/hyperseg), because that
# is where 02_train.sh and 03_eval.sh resolve their relative paths from.
#
# Usage: scripts/01_prepare_data.sh /path/to/low_altitude_2026 [work_dir]
set -euo pipefail

DATA="${1:?usage: scripts/01_prepare_data.sh /path/to/low_altitude_2026 [work_dir]}"
WORK="${2:-$(cd "$(dirname "$0")/../hyperseg" && pwd)/experiment/round2_reproduce}"

for split in images masks; do
    if [ ! -d "$DATA/train/$split" ]; then
        echo "missing $DATA/train/$split" >&2
        exit 1
    fi
done

mkdir -p "$WORK/train/train"
ln -sfn "$(cd "$DATA/train/images" && pwd)" "$WORK/train/train/images"
ln -sfn "$(cd "$DATA/train/masks"  && pwd)" "$WORK/train/train/masks"

# The split files ship with the package; they are bare file-name lists.
ln -sfn "$(cd "$(dirname "$0")/../../06_数据说明/splits" && pwd)" "$WORK/splits" 2>/dev/null || true

echo "data root : $DATA"
echo "work dir  : $WORK"
for split in train val test; do
    if [ -f "$WORK/splits/$split.txt" ]; then
        printf '  %-5s %s ids\n' "$split" "$(wc -l < "$WORK/splits/$split.txt")"
    fi
done
echo "labelled images: $(find "$DATA/train/images" -name '*.png' | wc -l)"
echo "labelled masks : $(find "$DATA/train/masks"  -name '*.png' | wc -l)"
echo
echo "next: --data $WORK/train --split-dir $WORK/splits"
