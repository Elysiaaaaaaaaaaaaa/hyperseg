#!/usr/bin/env bash
set -u

root=/root/autodl-tmp/work_dirs/mathseg_fewshot_20260922
pid_file="$root/m3_loveda_resume.pid"
while test -s "$pid_file"; do
  pid=$(cat "$pid_file")
  test -d "/proc/$pid" || break
  sleep 30
done

base="$root/loveda/m3/semantic_head_2000"
for shots in 2 5 10; do
  source_dir="$base/$shots/semantic-head_${shots}shot_seed3407"
  target_dir="$base/${shots}shot_seed3407"
  if test -f "$source_dir/summary.json" && test ! -e "$target_dir"; then
    mv "$source_dir" "$target_dir"
  fi
done

printf '%s\n' "M3 LoveDA finalization finished" >> "$root/m3_loveda_resume.log"
