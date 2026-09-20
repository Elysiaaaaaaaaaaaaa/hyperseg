"""Poll server2 and archive H3 logs; keep this local process running until exit.

Uses the workspace's existing .tmp_server2_access.py connection helper. No
credentials are stored in this script. Interrupted downloads use .part files.
"""
import subprocess
import fcntl
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'h3_swin_l_hyperseg_160k_seed3407_fp32'
REMOTE = '/root/autodl-tmp/work_dirs/mask2former_uav/' + NAME
DEST = Path(__file__).resolve().parent / 'logs' / NAME
HELPER = ROOT / '.tmp_server2_access.py'


def access(*args):
    return subprocess.run([sys.executable, str(HELPER), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=90)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    lock = (DEST / 'sync.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('A log collector is already running', flush=True)
        return
    while True:
        try:
            status = access('exec', f'if test -f {REMOTE}/exit_code; then cat {REMOTE}/exit_code; else echo running; fi')
            if status.returncode:
                raise RuntimeError('SSH status check failed')
            complete = status.stdout.strip() in ('0', '1', '2', '137', '143')
            if status.stdout.strip().isdigit():
                complete = True
            success = True
            for name in ('train.log', 'config.json', 'val_metrics.json'):
                exists = access('exec', f'test -f {REMOTE}/{name}')
                if exists.returncode:
                    if name == 'train.log':
                        success = False
                    continue
                temporary = DEST / (name + '.part')
                result = access('download', f'{REMOTE}/{name}', str(temporary))
                if result.returncode == 0:
                    temporary.replace(DEST / name)
                else:
                    success = False
            print(time.strftime('%Y-%m-%d %H:%M:%S'), status.stdout.strip(),
                  'sync_ok=' + str(success), flush=True)
            if complete and success:
                (DEST / 'exit_code').write_text(status.stdout.strip() + '\n')
                return
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(time.strftime('%Y-%m-%d %H:%M:%S'), type(error).__name__, str(error), flush=True)
        time.sleep(300)


if __name__ == '__main__':
    main()
