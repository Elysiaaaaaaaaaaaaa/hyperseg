#!/usr/bin/env bash
# Copy the competition runtime onto AutoDL's system disk. Run on server2.
set -euo pipefail

source_root=/root/autodl-tmp
project_source=$source_root/hyperseg
data_source=$source_root/data/low_altitude_2026
project_target=/root/hyperseg
data_target=$project_target/dataset/low_altitude_2026
checkpoint_source=$source_root/work_dirs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt
checkpoint_target=$project_target/runs/mask2former_uav/h3_swin_l_hyperseg_160k_seed3407_fp32/best.pt

for path in "$project_source" "$data_source/train/images" "$data_source/train/masks" \
    "$data_source/images" "$data_source/test_2/images" "$checkpoint_source" \
    "$source_root/mmsegmentation"; do
    [[ -e "$path" ]] || { echo "Missing source: $path" >&2; exit 1; }
done

required_bytes=$(du -sb "$project_source" "$data_source/train" "$data_source/images" \
    "$data_source/test_2" "$checkpoint_source" "$source_root/mmsegmentation" \
    | awk '{sum += $1} END {printf "%.0f\n", sum}')
available_bytes=$(df -B1 --output=avail /root | tail -n 1 | tr -d ' ')
reserve_bytes=$((3 * 1024 * 1024 * 1024))
printf 'Required upper bound: %s bytes; system disk available: %s bytes\n' \
    "$required_bytes" "$available_bytes"
if (( available_bytes < required_bytes + reserve_bytes )) && [[ ! -d "$data_target/train" ]]; then
    echo 'System disk lacks space for first copy and 3 GiB reserve' >&2
    exit 1
fi

mkdir -p "$project_target" "$data_target" "$(dirname "$checkpoint_target")"
echo 'Copying project code, virtual environment, and model assets...'
rsync -a --exclude='/runs/' --exclude='__pycache__/' \
    "$project_source/" "$project_target/"
rsync -a "$source_root/mmsegmentation/" /root/mmsegmentation/
rsync -a "$project_source/runs/splits/" "$project_target/runs/splits/"

# Console scripts and activation files contain absolute virtualenv paths.
old_venv="$project_source/.venv-mask2former"
new_venv="$project_target/.venv-mask2former"
while IFS= read -r -d '' file; do
    if grep -Iq "$old_venv" "$file"; then
        sed -i "s|$old_venv|$new_venv|g" "$file"
    fi
done < <(find "$new_venv/bin" -maxdepth 1 -type f -print0)
sed -i "s|$old_venv|$new_venv|g" "$new_venv/pyvenv.cfg"

echo 'Copying labeled competition data...'
rsync -a "$data_source/train/" "$data_target/train/"
echo 'Copying competition test images...'
rsync -a "$data_source/images/" "$data_target/images/"
rsync -a "$data_source/test_2/" "$data_target/test_2/"
for name in Label.txt stratified_split_90_10.json; do
    [[ ! -f "$data_source/$name" ]] || cp -a "$data_source/$name" "$data_target/$name"
done

echo 'Copying H3 best checkpoint and run metadata...'
rsync -a "$checkpoint_source" "$checkpoint_target"
for name in config.json best_test_metrics.json; do
    source_file=$(dirname "$checkpoint_source")/$name
    [[ ! -f "$source_file" ]] || cp -a "$source_file" "$(dirname "$checkpoint_target")/$name"
done
mkdir -p "$project_target/runs/mask2former_uav/h3_test2_20260926"
manifest_source=$source_root/work_dirs/mask2former_uav/h3_test2_20260926/data_manifest.json
if [[ -f "$manifest_source" ]]; then
    cp -a "$manifest_source" "$project_target/runs/mask2former_uav/h3_test2_20260926/data_manifest.json"
fi

echo 'Copy complete; sources remain on the data disk for verification.'
df -h /root
