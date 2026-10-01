#!/usr/bin/env python3
"""把 `h3_bgfix` 各臂的散落指标汇成一份可引用的表（纯 CPU，本地跑）。

数据来源（都已在本地，无需服务器）：

    logs/collected_20k/<arm>/val_history.jsonl   训练时的原生 val 曲线
    results/proxy_20k_all_arms.json              事后补跑的多尺度代理评估
    results/test2_compare_20k*.json              test_2 无标注预测的背景分布对照

test_2 那份会自动挑「覆盖臂最多」的文件：先跑的四臂在 `test2_compare_20k.json`，
补了 `loss01` 之后是 `test2_compare_20k_loss01.json`。要指定就用 `--test2-json`。

用法：

    python experiment/h3_bgfix_20260930/summarize_arms.py \
        --out experiment/h3_bgfix_20260930/results/arms_summary_20k.json

两个口径上的坑，脚本里已经处理：

* `val_metrics.json` 存的是**收官点**，不是 best 点。指标一律按 `selection_score`
  在 `val_history.jsonl` 上取最大（`baseline` 的 best 在 17000、`loss01` 在 19000）。
* 代理评估的键是完整检查点路径，臂名要从 `h3_bgfix_<arm>_20k_seed3407` 里剥出来，
  不能假设 JSON 键的顺序与命令行一致。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent
ARMS = ("baseline", "loss01", "loss", "aug", "full")
ARM_RE = re.compile(r"h3_bgfix_([a-z0-9]+)_\d+k_seed\d+")


def best_record(history: Path) -> dict | None:
    rows = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        return None
    return max(rows, key=lambda row: row["selection_score"])


def native_block(record: dict) -> dict:
    val = record["val"]
    return {
        "iter": record["iter"],
        "selection_score": record["selection_score"],
        "mIoU": val["mIoU"],
        "background_false_positive_rate": val["background_false_positive_rate"],
        "background_false_negative_rate": val["background_false_negative_rate"],
        "per_class_iou": {k: val["per_class_iou"][k] for k in sorted(val["per_class_iou"], key=int)},
    }


def proxy_block(payload: dict | None) -> dict | None:
    if not payload:
        return None
    metrics = payload["metrics"]
    scales = sorted((k for k in metrics if k.startswith("proxy@")),
                    key=lambda k: float(k.split("@")[1]))
    return {
        "step": payload.get("step"),
        "per_scale": {k.split("@")[1]: metrics[k]["mIoU"] for k in scales},
        "mean": payload.get("proxy_mIoU_mean"),
        "native_recomputed": payload.get("native_mIoU"),
        "native_minus_proxy_gap": payload.get("proxy_gap"),
    }


def pick_test2_json(explicit: Path | None, results: Path) -> Path | None:
    """挑一份覆盖臂最多的 test_2 对比 JSON；并列时取最新的。"""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"--test2-json 不存在: {explicit}")
        return explicit
    best: tuple[int, float, Path] | None = None
    for candidate in sorted(results.glob("test2_compare_20k*.json")):
        try:
            count = len(json.loads(candidate.read_text(encoding="utf-8")).get("sets", {}))
        except (OSError, ValueError):
            continue
        key = (count, candidate.stat().st_mtime, candidate)
        if best is None or key[:2] > best[:2]:
            best = key
    return best[2] if best else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="汇总 h3_bgfix 各臂指标")
    parser.add_argument("--logs", type=Path, default=EXPERIMENT / "logs" / "collected_20k")
    parser.add_argument("--results", type=Path, default=EXPERIMENT / "results")
    parser.add_argument("--test2-json", type=Path, default=None,
                        help="指定 test_2 对比 JSON；默认自动挑覆盖臂最多的那份")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    proxy_raw: dict[str, dict] = {}
    proxy_json = args.results / "proxy_20k_all_arms.json"
    if proxy_json.is_file():
        for checkpoint, payload in json.loads(proxy_json.read_text(encoding="utf-8"))["checkpoints"].items():
            match = ARM_RE.search(checkpoint)
            if match:
                proxy_raw[match.group(1)] = payload

    test2_raw: dict[str, dict] = {}
    test2_json = pick_test2_json(args.test2_json, args.results)
    if test2_json is not None:
        test2_raw = json.loads(test2_json.read_text(encoding="utf-8")).get("sets", {})

    summary: dict = {"source": {
        "logs": str(args.logs), "results": str(args.results),
        "proxy_json": proxy_json.name if proxy_json.is_file() else None,
        "test2_json": test2_json.name if test2_json.is_file() else None,
    }, "arms": {}, "notes": [
        "原生指标取 val_history.jsonl 中 selection_score 最大的那一行（即 best.pt 对应的点）。",
        "代理 = 有标注 val 699 张在 0.5/0.625/0.75 三个降采样尺度上的 mIoU 均值。",
        "test_2 无真值，只报背景占比分布与参考集 delta，不能当精度用。",
    ]}

    baseline_miou = None
    for arm in ARMS:
        entry: dict = {}
        record = None
        history = args.logs / arm / "val_history.jsonl"
        if history.is_file():
            record = best_record(history)
        if record:
            entry["native"] = native_block(record)
            if arm == "baseline":
                baseline_miou = entry["native"]["mIoU"]
        entry["proxy"] = proxy_block(proxy_raw.get(arm))
        if arm in test2_raw:
            dist = test2_raw[arm]["background_distribution"]
            entry["test2"] = {
                "background_median": dist["median"],
                "background_global_share": test2_raw[arm]["class_shares"]["1"],
                "images_over_0.8": dist["over_0.8"],
                "images_over_0.6": dist["over_0.6"],
                "delta_vs": (test2_raw[arm].get("delta_vs_reference") or {}).get("reference"),
                "left_collapsed_pool": (test2_raw[arm].get("delta_vs_reference") or {}).get("left_collapsed_pool"),
                "entered_collapsed_pool": (test2_raw[arm].get("delta_vs_reference") or {}).get("entered_collapsed_pool"),
            }
        summary["arms"][arm] = entry

    if baseline_miou:
        for entry in summary["arms"].values():
            if entry.get("native"):
                entry["native"]["mIoU_delta_pts_vs_baseline"] = (
                    entry["native"]["mIoU"] - baseline_miou) * 100

    baseline_proxy = (summary["arms"]["baseline"].get("proxy") or {}).get("mean")
    if baseline_proxy:
        for entry in summary["arms"].values():
            if entry.get("proxy"):
                entry["proxy"]["mean_delta_pts_vs_baseline"] = (
                    entry["proxy"]["mean"] - baseline_proxy) * 100

    header = (f"{'arm':<9}{'best':>7}{'native':>9}{'ΔmIoU':>8}{'FPR':>8}{'FNR':>8}"
              f"{'proxy':>9}{'Δproxy':>8}{'>0.8':>7}")
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        entry = summary["arms"][arm]
        nat, prox = entry.get("native"), entry.get("proxy")
        t2 = entry.get("test2") or {}

        def fmt(value, spec, width, suffix=""):
            if value is None:
                return f"{'—':>{width}}"
            return f"{value:{spec}}{suffix}"

        native_delta = nat.get("mIoU_delta_pts_vs_baseline") if nat else None
        proxy_delta = prox.get("mean_delta_pts_vs_baseline") if prox else None
        print(f"{arm:<9}"
              f"{fmt(nat['iter'] if nat else None, '>7', 7)}"
              f"{fmt(nat['mIoU'] if nat else None, '>9.4f', 9)}"
              f"{fmt(native_delta, '>+8.2f', 8)}"
              f"{fmt(nat['background_false_positive_rate'] * 100 if nat else None, '>7.2f', 7, '%')}"
              f"{fmt(nat['background_false_negative_rate'] * 100 if nat else None, '>7.2f', 7, '%')}"
              f"{fmt(prox['mean'] if prox else None, '>9.4f', 9)}"
              f"{fmt(proxy_delta, '>+8.2f', 8)}"
              f"{fmt(t2.get('images_over_0.8'), '>7', 7)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
