"""Pull and summarise the H3 bgfix screening arms.

    python experiment/h3_bgfix_20260930/summarize_screen.py --arms baseline loss aug full

Reads ``val_history.jsonl`` from each arm's work directory on the server, caches it under
``results/screen_<suffix>/``, then prints the per-arm trajectories and the side-by-side
comparison the attribution argument needs.  Safe to run while the arms are still training --
it reports how far each arm has got and does not treat a partial run as a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"

RELEVANT_CONFIG = (
    "preset", "aug_mode", "false_bg_weight", "class_weight_mode", "proxy_weight",
    "scale_min", "scale_max", "gsd_prob", "photo_prob", "scene_crop_prob",
    "max_iters", "val_interval", "seed",
)


def remote_work_dir(arm: str, suffix: str) -> str:
    return f"/root/hyperseg/runs/mask2former_uav/h3_bgfix_{arm}_{suffix}_seed3407"


def fetch(client, module, arm: str, suffix: str, cache: Path):
    """Return ``(history, config, finished)`` for one arm, caching the raw files locally."""
    work_dir = remote_work_dir(arm, suffix)
    cache.mkdir(parents=True, exist_ok=True)

    _, history_text = module.run(client, f"cat {work_dir}/val_history.jsonl 2>/dev/null || true")
    _, config_text = module.run(client, f"cat {work_dir}/config.json 2>/dev/null || true")
    _, exit_text = module.run(client, f"cat {work_dir}/exit_code 2>/dev/null || true")

    (cache / f"{arm}_val_history.jsonl").write_text(history_text, encoding="utf-8")
    if config_text.strip():
        (cache / f"{arm}_config.json").write_text(config_text, encoding="utf-8")

    history = []
    for line in history_text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            history.append(json.loads(line))
    exit_code = exit_text.strip().splitlines()
    finished = bool(exit_code)
    return history, (json.loads(config_text) if config_text.strip().startswith("{") else {}), finished, exit_code


def show_trajectory(arm: str, history: list[dict], finished: bool) -> None:
    print(f"\n=== {arm} ===  {'finished' if finished else f'in progress, {len(history)} validations'}")
    header = f"{'iter':>7} {'native mIoU':>12} {'bg FPR':>8} {'bg FNR':>8} {'proxy mIoU':>11} {'select':>8}"
    print(header)
    for record in history:
        native = record.get("val", {})
        proxy = record.get("proxy_mIoU_mean")
        print(f"{record['iter']:>7} {native.get('mIoU', float('nan')):>12.4f} "
              f"{native.get('background_false_positive_rate', float('nan')):>8.4f} "
              f"{native.get('background_false_negative_rate', float('nan')):>8.4f} "
              f"{proxy if proxy is not None else float('nan'):>11.4f} "
              f"{record.get('selection_score', float('nan')):>8.4f}")


def best_native(history: list[dict]):
    """The checkpoint that native-mIoU selection would have kept."""
    scored = [(record["val"]["mIoU"], record) for record in history if "val" in record]
    return max(scored, key=lambda item: item[0]) if scored else (float("nan"), {})


def best_selected(history: list[dict]):
    """The checkpoint the run's own selection score kept (proxy-aware for the full arm)."""
    scored = [(record.get("selection_score", float("-inf")), record) for record in history]
    return max(scored, key=lambda item: item[0]) if scored else (float("nan"), {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="server2", choices=("server1", "server2", "server3"))
    parser.add_argument("--suffix", default="20k", help="budget suffix used in the work directory name")
    parser.add_argument("--arms", nargs="+", default=["baseline", "loss", "aug", "full"])
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    import remote  # noqa: E402

    client = remote.connect(args.server)
    cache = RESULTS / f"screen_{args.suffix}"
    try:
        arms = {}
        for arm in args.arms:
            history, config, finished, exit_code = fetch(client, remote, arm, args.suffix, cache)
            arms[arm] = {"history": history, "config": config, "finished": finished, "exit_code": exit_code}
    finally:
        client.close()

    for arm in args.arms:
        record = arms[arm]
        show_trajectory(arm, record["history"], record["finished"])
        if record["exit_code"] and record["exit_code"][0] != "0":
            print(f"  !! exited with status {record['exit_code'][0]} -- treat this arm as invalid")

    print("\n" + "=" * 74)
    print("per-arm best, under native-mIoU selection and under the run's own selection score")
    print("=" * 74)
    print(f"{'arm':>9} {'preset':>9} {'aug':>7} {'fbg':>5} {'cw':>6} {'best-native':>12} "
          f"{'at iter':>8} {'bgFPR':>7} {'bgFNR':>7} {'best-select':>12}")
    for arm in args.arms:
        record = arms[arm]
        history, config = record["history"], record["config"]
        if not history:
            print(f"{arm:>9}  (no data yet)")
            continue
        native_score, native_record = best_native(history)
        select_score, select_record = best_selected(history)
        native_val = native_record.get("val", {})
        print(f"{arm:>9} {config.get('preset', '?'):>9} {config.get('aug_mode', '?'):>7} "
              f"{config.get('false_bg_weight', '?'):>5} {config.get('class_weight_mode', '?'):>6} "
              f"{native_score:>12.4f} {native_record.get('iter', 0):>8} "
              f"{native_val.get('background_false_positive_rate', float('nan')):>7.4f} "
              f"{native_val.get('background_false_negative_rate', float('nan')):>7.4f} "
              f"{select_score:>12.4f}")

    # Whatever the arms have validated in common, so the comparison is like-for-like.
    common = set.intersection(*[{record["iter"] for record in arms[arm]["history"]} for arm in args.arms
                                if arms[arm]["history"]]) if all(arms[arm]["history"] for arm in args.arms) else set()
    if common:
        print("\nat the last iteration every arm has reached")
        print(f"{'iter':>7} " + " ".join(f"{arm:>10}" for arm in args.arms))
        for metric in ("mIoU", "background_false_positive_rate", "background_false_negative_rate"):
            row = []
            for arm in args.arms:
                value = next((r["val"].get(metric) for r in arms[arm]["history"] if r["iter"] == max(common)), None)
                row.append(f"{value:.4f}" if value is not None else "n/a")
            print(f"{max(common):>7} " + " ".join(f"{cell:>10}" for cell in row) + f"   <- {metric}")
    else:
        print("\nnot every arm has reached the same iteration yet; no like-for-like table")

    print(f"\nraw files cached in {cache}")
    print("next: evaluate each arm's best.pt on the reduced-detail proxy set and on test_2 "
          "(regression_test2.py), then judge the four pre-registered criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
