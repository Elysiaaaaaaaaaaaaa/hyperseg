"""Upload a checkpoint (or any large file) to a server, then verify its sha256.

Large transfers are the one part of a remote inference run that can silently
half-finish, so this prints throughput while it runs and finishes by asking the
server for the digest of what actually landed.  Credentials handling is imported
from the h3_bgfix experiment to keep a single parser for ``sshconfig.toml``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiment/h3_bgfix_20260930'))

import remote  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default='server2')
    parser.add_argument('--local', required=True, help='local file to send')
    parser.add_argument('--remote', required=True, help='absolute destination path')
    args = parser.parse_args()

    local = Path(args.local).resolve()
    if not local.is_file():
        raise SystemExit(f'no such local file: {local}')
    size = local.stat().st_size
    print(f'local : {local}\n        {size} bytes', flush=True)

    client = remote.connect(args.server)
    try:
        sftp = client.open_sftp()
        try:
            existing = sftp.stat(args.remote)
            print(f'note  : remote path already exists ({existing.st_size} bytes), overwriting',
                  flush=True)
        except IOError:
            pass

        started = time.monotonic()
        last = [0.0]

        def report(transferred: int, total: int) -> None:
            now = time.monotonic()
            if transferred < total and now - last[0] < 5:
                return
            last[0] = now
            elapsed = max(now - started, 1e-6)
            print(f'  {transferred / 2**20:9.1f} / {total / 2**20:.1f} MiB'
                  f'   {transferred / elapsed / 2**20:5.2f} MiB/s', flush=True)

        sftp.put(str(local), args.remote, callback=report)
        elapsed = time.monotonic() - started
        landed = sftp.stat(args.remote).st_size
        print(f'done  : {landed} bytes in {elapsed:.0f}s '
              f'({landed / max(elapsed, 1e-6) / 2**20:.2f} MiB/s average)', flush=True)
        if landed != size:
            raise SystemExit(f'size mismatch: local {size} vs remote {landed}')

        _, output = remote.run(client, f'sha256sum {args.remote}')
        print(f'remote sha256: {output.strip()}', flush=True)
        return 0
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
