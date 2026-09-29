"""Sequential server queue for M0-M3 x SUIM/LoveDA fixed few-shot grids."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path('/root/autodl-tmp'))
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = json.loads((output / "preflight.json").read_text())
    assert len(preflight['checks']) == 8
    jobs = []
    for variant in ("m0", "m1", "m2", "m3"):
        for dataset, script, data, manifest in (
            ("suim", "fewshot_grid.py", "SUIM", "runs/suim_manual/export_20260913_145956_961191"),
            ("loveda", "fewshot_loveda_grid.py", "LoveDA", "runs/loveda_manual/export_20260912_101105_509451"),
        ):
            command = [sys.executable, '-u', str(ROOT / 'experiment/mathseg_uav' / script),
                '--data-root', str(args.data_root / data), '--manifest-dir', str(ROOT / manifest),
                '--init-checkpoint', str(args.source_root / f'{variant}_seed3407/best.pt'),
                '--output-root', str(output / dataset / variant), '--checkpoint-format', 'compact',
                '--shots', '1', '2', '5', '10', '--seeds', *map(str, args.seeds),
                '--num-workers', '4', '--amp', '--device', 'cuda', '--skip-completed']
            jobs.append(dict(variant=variant, dataset=dataset, command=command))
    (output / 'plan.json').write_text(json.dumps(dict(jobs=jobs, total_runs=128 * len(args.seeds)), indent=2)+'\n')
    status = dict(state='running', pid=os.getpid(), started=time.time(), completed_grids=[], total_runs=128 * len(args.seeds))
    def save():
        temporary = output / 'queue_status.tmp'
        temporary.write_text(json.dumps(status, indent=2)+'\n')
        temporary.replace(output / 'queue_status.json')
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for job in jobs:
        status['current'] = {key: job[key] for key in ('variant', 'dataset')}
        save()
        name = f"{job['variant']}_{job['dataset']}"
        print(f'START {name}', flush=True)
        with (output / f'{name}.log').open('a') as log:
            result = subprocess.run(job['command'], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state='failed', exit_code=result.returncode, finished=time.time())
            save()
            raise SystemExit(result.returncode)
        status['completed_grids'].append(name)
        save()
        print(f'DONE {name}', flush=True)
    status.update(state='completed', finished=time.time())
    save()
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
