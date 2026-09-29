"""Download server2 few-shot logs and maintain the experiment's progress analysis."""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/root/autodl-tmp/work_dirs/mathseg_fewshot_20260922'
LOCAL = ROOT / 'experiment/mathseg_uav/logs/fewshot_20260922'
START = '<!-- FEWSHOT_20260922_START -->'
END = '<!-- FEWSHOT_20260922_END -->'


def access(*args):
    result = subprocess.run([sys.executable, str(ROOT / '.tmp_server2_access.py'), *args],
                            cwd=ROOT, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout


def planned_seeds(status, plan):
    """Read the active queue plan; historical results must not expand its seeds."""
    if status.get('planned_seeds'):
        return sorted(set(status['planned_seeds']))
    seeds = set()
    for job in plan.get('jobs', []):
        command = job['command']
        if '--seeds' in command:
            for value in command[command.index('--seeds') + 1:]:
                if value.startswith('--'):
                    break
                seeds.add(int(value))
    if not seeds:
        raise ValueError('Cannot establish planned seeds from queue status or plan')
    return sorted(seeds)


def sync():
    archive = '/tmp/mathseg_fewshot_logs_20260922.tgz'
    # tar exit 1 only means a live log grew while being archived; retry next cycle.
    access('exec', f'cd {REMOTE} && tar --exclude="*.pt" --exclude="*.tmp" -czf {archive} .; code=$?; test "$code" -le 1')
    LOCAL.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mathseg_sync_') as temporary:
        path = Path(temporary) / 'logs.tgz'
        access('download', archive, str(path))
        extracted = Path(temporary) / 'extracted'
        extracted.mkdir()
        with tarfile.open(path) as bundle:
            bundle.extractall(extracted, filter='data')
        subprocess.run(['rsync', '-a', str(extracted) + '/', str(LOCAL) + '/'], check=True)
    status = json.loads((LOCAL / 'queue_status.json').read_text())
    plan = json.loads((LOCAL / 'plan.json').read_text())
    seeds = planned_seeds(status, plan)
    health_command = """/root/miniconda3/bin/python - <<'PY'
import json, pathlib, subprocess, datetime
root = pathlib.Path('/root/autodl-tmp/work_dirs/mathseg_fewshot_20260922')
status = json.loads((root / 'queue_status.json').read_text())
cmdline = pathlib.Path('/proc') / str(status['pid']) / 'cmdline'
alive = cmdline.exists() and b'queue_fewshot.py' in cmdline.read_bytes()
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader'], capture_output=True, text=True)
print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), queue_alive=alive,
                     gpu_exit_code=gpu.returncode, gpu_output=gpu.stdout.strip(), gpu_error=gpu.stderr.strip())))
PY"""
    health = json.loads(access('exec', health_command))
    (LOCAL / 'server_health.json').write_text(json.dumps(health, indent=2)+'\n')
    observed_state = status['state']
    if observed_state == 'running' and not health['queue_alive']:
        observed_state = 'interrupted (queue process absent)'
    results = []
    for path in LOCAL.glob('*/*/*/*/**/summary.json'):
        data = json.loads(path.read_text())
        parts = path.relative_to(LOCAL).parts
        results.append((parts[0], parts[1], parts[2], data))
    all_completed = len(results)
    results = [row for row in results if row[3]["seed"] in seeds]
    total = status.get("revised_total_runs", status["total_runs"])
    expected = {(d, v, setting, k, seed)
                for d in ('suim', 'loveda') for v in ('m0', 'm1', 'm2', 'm3')
                for setting in ('adapter_200', 'adapter_2000', 'semantic_head_200', 'semantic_head_2000')
                for k in (1, 2, 5, 10) for seed in seeds}
    identities = [(d, v, setting, r['shots_per_class'], r['seed']) for d, v, setting, r in results]
    if len(identities) == len(expected) and set(identities) == expected:
        observed_state = 'completed (all planned result files present)'
    rows = {}
    for dataset, variant, setting, result in results:
        key = (dataset, variant, setting, result['shots_per_class'])
        rows.setdefault(key, []).append(result['metrics']['mIoU'])
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = [START, '## 2026-09-22 few-shot 服务器运行追踪', '',
        f'更新 UTC：{timestamp}；实测队列状态：{observed_state}；当前计划 seed={seeds}，完成 {len(results)}/{total} 次，历史全部种子共完成 {all_completed} 次。',
        f'队列最后记录组：{status.get("current")}；原队列状态仅作历史记录，完成情况以结果文件为准。本节由 sync_fewshot_logs.py 自动维护。', '',
        'LoveDA：原始 1..7→0..6，原始 0/255→255 Ignore，7 通道；SUIM：8 类全部有效。',
        '同一数据集复用固定支持集；最终步评估，不使用评估集选 checkpoint。',
        '下表按当前计划的种子汇总；单种子结果不代表跨种子稳定性。', '',
        '| 数据集 | 模型 | 设置 | K | seeds | mIoU % | 标准差 pp | 相对 M0 pp |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for (dataset, variant, setting, k), values in sorted(rows.items()):
        baseline = rows.get((dataset, 'm0', setting, k), [])
        delta = f'{(statistics.mean(values)-statistics.mean(baseline))*100:+.4f}' if len(values)==len(baseline)==len(seeds) else '待齐全'
        std = f'{statistics.stdev(values)*100:.4f}' if len(values)>1 else '—'
        lines.append(f'| {dataset} | {variant} | {setting} | {k} | {len(values)} | {statistics.mean(values)*100:.4f} | {std} | {delta} |')
    if status['state'] == 'failed':
        lines += ['', f'队列失败，退出码 {status.get("exit_code")}。请查对应组日志；后续组未启动。']
    lines += ['', '日志和逐次指标：`logs/fewshot_20260922/`。压缩权重留在服务器，源 UAV checkpoint 必须保留。', END]
    report = '\n'.join(lines)+'\n'
    (LOCAL / 'analysis.md').write_text(report, encoding='utf-8')
    document = ROOT / 'experiment/mathseg_uav/experiment.md'
    text = document.read_text(encoding='utf-8')
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + report + '\n' + after.lstrip('\n')
    else:
        text += '\n\n' + report
    document.write_text(text, encoding='utf-8')
    print(timestamp, observed_state, f'{len(results)}/{total}', flush=True)
    return status['state'] in ('completed', 'failed', 'paused') or not health['queue_alive']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--stop-background', action='store_true')
    parser.add_argument('--interval', type=int, default=600)
    args = parser.parse_args()
    if args.stop_background and (LOCAL / 'sync.pid').exists():
        pid = int((LOCAL / 'sync.pid').read_text())
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
            if str(Path(__file__).resolve()).encode() in command and b'--interval' in command:
                os.kill(pid, signal.SIGTERM)
                print(f'Stopped background sync PID {pid}', flush=True)
        except (FileNotFoundError, ProcessLookupError):
            pass
    if args.background:
        LOCAL.mkdir(parents=True, exist_ok=True)
        with (LOCAL / 'sync.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--interval', str(args.interval)], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (LOCAL / 'sync.pid').write_text(str(process.pid)+'\n')
        print(f'Background sync PID {process.pid}', flush=True)
        return
    while True:
        try:
            finished = sync()
            if args.once or finished:
                return
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
