#!/usr/bin/env bash
# Inference-then-train driver for the evening of 2026-09-30.
#
#   nohup bash experiment/h3_bgfix_20260930/run_infer_test2_then_loss01.sh \
#       >/dev/null 2>&1 < /dev/null &
#
# Two jobs that both need the only GPU, so they are chained rather than overlapped: the
# card is already at 96% utilisation with a single training process, so running inference
# alongside a training arm would slow both without finishing either sooner.
#
# Job 1: export test_2 predictions for all four 20k arms, using the published H3 protocol
#        (512 px window, 0.5 overlap) so the masks are directly comparable with the
#        existing `swin_l_hyperseg_复赛.zip` baseline.  Each arm's PNG directory is packed
#        into a tarball once it reaches the full 1300 files.
# Job 2: start the `false_bg_weight=0.1` arm.  It is the `loss` preset with exactly one
#        knob changed, so the dose-response of the false-background penalty is isolated
#        from the augmentation change.
#
# Every step appends to one log, so a single download explains the whole evening.
set -uo pipefail

project_root=/root/hyperseg
cd "$project_root"

runs="$project_root/runs/mask2former_uav"
images="$project_root/dataset/low_altitude_2026/test_2/images"
python_bin="$project_root/.venv-mask2former/bin/python"
log="$runs/h3_bgfix_infer_test2.log"

export PYTHONPATH="$project_root/../mmsegmentation:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

say() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$log"; }

: > "$log"
say "inference driver start: $(ls "$images"/*.png | wc -l) images, 512 px window, 0.5 overlap"
say "free before: $(df -h /root | tail -1 | awk '{print $4}')"

expected=1300
infer_failed=0

for arm in baseline loss aug full; do
    checkpoint="$runs/h3_bgfix_${arm}_20k_seed3407/best.pt"
    output="$runs/pred_test2_${arm}_20k"

    if [[ ! -f "$checkpoint" ]]; then
        say "SKIP $arm: checkpoint missing at $checkpoint"
        infer_failed=1
        continue
    fi

    rm -rf "$output"
    mkdir -p "$output"
    started=$(date +%s)
    say "START $arm"

    "$python_bin" -u experiment/mask2former_uav/infer_h3_hyperseg.py \
        --checkpoint "$checkpoint" \
        --input "$images" \
        --output "$output" \
        --size 512 --overlap 0.5 >> "$log" 2>&1
    code=$?

    written=$(ls "$output"/*.png 2>/dev/null | wc -l)
    elapsed=$(( $(date +%s) - started ))
    say "DONE $arm exit=$code png=$written elapsed=${elapsed}s"

    if [[ "$code" -eq 0 && "$written" -eq "$expected" ]]; then
        # tar rather than zip: the PNGs are already deflated, so compression buys little and
        # `tar -cf` on a big directory is the one thing that reliably exists on these boxes.
        tar -czf "$runs/pred_test2_${arm}_20k.tar.gz" -C "$runs" "pred_test2_${arm}_20k" \
            && say "packed pred_test2_${arm}_20k.tar.gz ($(du -h "$runs/pred_test2_${arm}_20k.tar.gz" | cut -f1))"
    else
        say "WARNING $arm did not produce $expected predictions (exit=$code, got=$written); not packed"
        infer_failed=1
    fi
    say "free after $arm: $(df -h /root | tail -1 | awk '{print $4}')"
done

say "inference stage finished (failures=$infer_failed)"
say "--- launching loss01 training (preset=loss, false_bg_weight=0.1) ---"

# The launcher performs its own preflight, smoke test and CUDA check, and detaches the
# trainer itself, so this call returns once the run is live.
bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh loss loss01_20k \
    --max-iters 20000 --val-interval 1000 --drop-optimizer-state --false-bg-weight 0.1 \
    >> "$log" 2>&1
say "loss01 launcher returned exit=$?"

work="$runs/h3_bgfix_loss01_20k_seed3407"
if [[ -f "$work/train.pid" ]]; then
    say "loss01 training pid=$(head -n1 "$work/train.pid")"
else
    say "WARNING: loss01 launcher produced no train.pid; inspect $work/preflight.log"
fi
say "driver done"
