"""Remote channel for the H3 bgfix experiment (paramiko, no sshpass needed).

The project's other skills assume ``sshpass``; it is not installed on this machine, so this
module talks to the AutoDL boxes over paramiko instead.  Credentials are read from
``sshconfig.toml`` at the repository root and never printed.

Commands::

    python experiment/h3_bgfix_20260930/remote.py check
    python experiment/h3_bgfix_20260930/remote.py sync
    python experiment/h3_bgfix_20260930/remote.py run  -- "bash experiment/h3_bgfix_20260930/launch_h3_bgfix.sh full"
    python experiment/h3_bgfix_20260930/remote.py put  --local <file> --remote <path>
    python experiment/h3_bgfix_20260930/remote.py pull --remote <path> [<path> ...] --local <dir>
    python experiment/h3_bgfix_20260930/remote.py collect --suffix 20k --local experiment/h3_bgfix_20260930/logs/collected_20k
    python experiment/h3_bgfix_20260930/remote.py tail --remote <path> --lines 60

``sync`` uploads this experiment directory; nothing outside
``experiment/h3_bgfix_20260930/`` is modified by the fix, so that is all that has to move.
Use ``put`` instead of ``sync`` while a slot driver is running -- see ``command_put``.
``check`` verifies the pieces the fix depends on and are easy to get wrong: an actually
running GPU, the sibling MMSegmentation checkout on PYTHONPATH, the Swin-L backbone
checkpoint, the dataset, and the hashes of the two files imported from elsewhere in the
repo (``hyperseg_uav/swin_l_model.py`` and ``infer_h3_hyperseg.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/hyperseg"

# Files the fix imports from elsewhere in the repository.  A mismatch means the server is
# running different code from the local checkout.
DEPENDENCIES = [
    "hyperseg_uav/swin_l_model.py",
    "hyperseg_uav/model.py",
    "hyperseg_uav/__init__.py",
    "experiment/mask2former_uav/infer_h3_hyperseg.py",
]


def load_credentials(server: str = "server3"):
    config = ROOT / "sshconfig.toml"
    if not config.is_file():
        raise FileNotFoundError(config)
    text = config.read_text(encoding="utf-8")
    match = re.search(r"\[" + re.escape(server) + r"\](.*?)(?=\n\[|\Z)", text, re.S)
    if not match:
        raise KeyError(f"{server} not found in sshconfig.toml")
    block = match.group(1)
    port = re.search(r"-p\s+(\d+)", block)
    host = re.search(r"root@([\w\.\-]+)", block)
    password = re.search(r"password\s*=\s*(\S+)", block)
    if not (port and host and password):
        raise ValueError(f"incomplete entry for {server}")
    return host.group(1), int(port.group(1)), password.group(1)


def connect(server: str = "server3", timeout: int = 20):
    import paramiko

    host, port, password = load_credentials(server)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, port=port, username="root", password=password,
                       timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
    except paramiko.ssh_exception.NoValidConnectionsError as error:
        raise SystemExit(
            f"cannot reach {server} at {host}:{port} -- {error}\n"
            "All three AutoDL boxes refuse connections while powered off, and outbound "
            "internet works, so this almost always means the instance is shut down.\n"
            "Power it on in the AutoDL console, wait for the status to become 'running', "
            "then re-run this command."
        ) from error
    except paramiko.ssh_exception.AuthenticationException as error:
        raise SystemExit(
            f"authentication rejected by {server}: {error}\n"
            "The password in sshconfig.toml may be stale; AutoDL rotates it on rebuild."
        ) from error
    return client


def run(client, command: str, timeout: int = 300) -> tuple[int, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace") + stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), output


def local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_check(args):
    client = connect(args.server)
    try:
        print(f"connected to {args.server}\n")
        print("== GPU ==")
        _, output = run(client, "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
                                "--format=csv,noheader || echo NO_GPU")
        print(output.strip() or "(no output)")

        print("\n== mmsegmentation checkout ==")
        _, output = run(client, f"ls -d {ROOT.name}/../mmsegmentation 2>/dev/null || "
                                f"ls -d /root/mmsegmentation 2>/dev/null || echo MISSING")
        print(output.strip())

        print("\n== project tree ==")
        for path in (f"{REMOTE_ROOT}/hyperseg_uav/swin_l_model.py",
                     f"{REMOTE_ROOT}/experiment/mask2former_uav/infer_h3_hyperseg.py",
                     f"{REMOTE_ROOT}/.venv-mask2former/bin/python",
                     f"{REMOTE_ROOT}/models/swin_l/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth",
                     f"{REMOTE_ROOT}/runs/splits/train.txt",
                     f"{REMOTE_ROOT}/dataset/low_altitude_2026/train/images",
                     f"{REMOTE_ROOT}/dataset/low_altitude_2026/test_2/images"):
            _, output = run(client, f"if [ -e {path} ]; then echo OK $(du -sh {path} 2>/dev/null | cut -f1); "
                                    f"else echo MISSING; fi")
            print(f"  {path}  ->  {output.strip()}")

        print("\n== dataset counts ==")
        _, output = run(client, f"ls {REMOTE_ROOT}/dataset/low_altitude_2026/train/images/*.png | wc -l; "
                                f"ls {REMOTE_ROOT}/dataset/low_altitude_2026/test_2/images/*.png | wc -l; "
                                f"wc -l {REMOTE_ROOT}/runs/splits/train.txt {REMOTE_ROOT}/runs/splits/val.txt")
        print(output.strip())

        print("\n== dependency hashes (remote vs local) ==")
        mismatch = 0
        for relative in DEPENDENCIES:
            local = ROOT / relative
            _, output = run(client, f"sha256sum {REMOTE_ROOT}/{relative} 2>/dev/null | cut -d' ' -f1")
            remote = output.strip()
            expected = local_sha256(local) if local.is_file() else "LOCAL_MISSING"
            flag = "OK" if remote == expected else "MISMATCH"
            if remote != expected:
                mismatch += 1
            print(f"  [{flag}] {relative}\n        remote {remote or 'MISSING'}\n        local  {expected}")

        print("\n== python environment ==")
        _, output = run(client, f"{REMOTE_ROOT}/.venv-mask2former/bin/python -c "
                                f"'import torch,mmseg;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -3")
        print(output.strip())

        print(f"\n{dependency_verdict(mismatch)}")
        return 0
    finally:
        client.close()


def dependency_verdict(mismatch: int) -> str:
    if mismatch:
        return (f"VERDICT: {mismatch} dependency file(s) differ -- re-upload them before "
                f"trusting a comparison against the published H3 result")
    return "VERDICT: dependency files match the local checkout"


def command_sync(args):
    import paramiko

    client = connect(args.server)
    try:
        sftp = client.open_sftp()

        def ensure(remote_dir: str):
            parts = remote_dir.strip("/").split("/")
            current = ""
            for part in parts:
                current += "/" + part
                try:
                    sftp.stat(current)
                except IOError:
                    sftp.mkdir(current)

        ensure(f"{REMOTE_ROOT}/{HERE.relative_to(ROOT).as_posix()}")
        uploaded = 0
        for path in sorted(HERE.rglob("*")):
            if not path.is_file():
                continue
            if any(part in {"__pycache__", "logs", "results"} for part in path.relative_to(HERE).parts):
                continue
            if path.suffix == ".pyc":
                continue
            relative = path.relative_to(ROOT).as_posix()
            remote = f"{REMOTE_ROOT}/{relative}"
            ensure(str(Path(remote).parent.as_posix()))
            sftp.put(str(path), remote)
            uploaded += 1
            print(f"  {relative}")
        print(f"\nuploaded {uploaded} file(s) to {REMOTE_ROOT}/{HERE.relative_to(ROOT).as_posix()}")
        _, output = run(client, f"chmod +x {REMOTE_ROOT}/experiment/h3_bgfix_20260930/*.sh; "
                                f"ls -la {REMOTE_ROOT}/experiment/h3_bgfix_20260930/ | head -30")
        print(output)
        return 0
    finally:
        client.close()


def command_run(args):
    client = connect(args.server)
    try:
        command = " ".join(args.command).lstrip("-").strip()
        if not command:
            raise SystemExit("nothing to run; use: remote.py run -- \"<command>\"")
        print(f"$ cd {args.cwd} && {command}\n")
        _, stdout, stderr = client.exec_command(f"cd {args.cwd} && {command}", timeout=args.timeout)
        for line in iter(stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
        error = stderr.read().decode("utf-8", "replace")
        if error.strip():
            print("--- stderr ---")
            print(error)
        code = stdout.channel.recv_exit_status()
        print(f"\nexit_code={code}")
        return code
    finally:
        client.close()


def command_pull(args):
    import paramiko

    client = connect(args.server)
    try:
        sftp = client.open_sftp()
        local_dir = Path(args.local)
        local_dir.mkdir(parents=True, exist_ok=True)
        for remote in args.remote:
            name = Path(remote).name
            target = local_dir / name
            sftp.get(remote, str(target))
            print(f"  {remote} -> {target} ({target.stat().st_size} bytes)")
        return 0
    finally:
        client.close()


def command_put(args):
    """Upload a single file.

    ``sync`` rewrites the whole experiment directory, which is wrong while the slot driver is
    running: bash reads a script as it executes it, so replacing ``screen_parallel_h3_bgfix.sh``
    underneath a live instance corrupts its position in the file.  This uploads exactly one path.
    """
    import paramiko

    client = connect(args.server)
    try:
        sftp = client.open_sftp()
        local = Path(args.local).resolve()
        if not local.is_file():
            raise SystemExit(f"no such local file: {local}")
        remote = args.remote if args.remote.startswith("/") else f"{REMOTE_ROOT}/{args.remote}"
        directory = str(Path(remote).parent).replace("\\", "/")
        current = ""
        for part in directory.strip("/").split("/"):
            current += "/" + part
            try:
                sftp.stat(current)
            except IOError:
                sftp.mkdir(current)
        sftp.put(str(local), remote)
        print(f"  {local} -> {remote} ({local.stat().st_size} bytes)")
        return 0
    finally:
        client.close()


def command_collect(args):
    """Download every log and metric produced by a screening run.

    Layout under ``--local``: one subdirectory per arm holding that arm's own artifacts, plus the
    slot driver's log, its nohup output, and the server-side progress log next to them.  A missing
    file is reported rather than aborting, because a partially finished run is still worth having.
    """
    import paramiko

    runs = f"{REMOTE_ROOT}/runs/mask2former_uav"
    target = Path(args.local)
    target.mkdir(parents=True, exist_ok=True)

    per_arm = ("train.log", "val_history.jsonl", "val_metrics.json", "config.json",
               "exit_code", "preflight.log", "smoke_test.log")
    wanted: list[tuple[str, Path]] = []
    for arm in args.arms:
        work_dir = f"{runs}/h3_bgfix_{arm}_{args.suffix}_seed3407"
        for name in per_arm:
            wanted.append((f"{work_dir}/{name}", target / arm / name))
    for name in ("h3_bgfix_screen.log", "h3_bgfix_screen.nohup.log", "h3_bgfix_watch.log"):
        wanted.append((f"{runs}/{name}", target / name))

    client = connect(args.server)
    try:
        sftp = client.open_sftp()
        got = missing = 0
        for remote, local in wanted:
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                sftp.get(remote, str(local))
            except IOError:
                missing += 1
                print(f"  [absent] {remote}")
                continue
            got += 1
            print(f"  {remote} -> {local.relative_to(target)} ({local.stat().st_size} bytes)")
        print(f"\ndownloaded {got} file(s) to {target} ({missing} absent)")
        return 0
    finally:
        client.close()


def command_tail(args):
    client = connect(args.server)
    try:
        _, output = run(client, f"tail -n {args.lines} {args.remote}")
        print(output)
        return 0
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="server3", choices=("server1", "server2", "server3"))
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("check", help="preflight the remote environment")

    subparsers.add_parser("sync", help="upload this experiment directory")

    run_parser = subparsers.add_parser("run", help="execute a command on the server")
    run_parser.add_argument("--cwd", default=REMOTE_ROOT)
    run_parser.add_argument("--timeout", type=int, default=3600)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    pull_parser = subparsers.add_parser("pull", help="download files")
    pull_parser.add_argument("--remote", nargs="+", required=True)
    pull_parser.add_argument("--local", required=True)

    put_parser = subparsers.add_parser("put", help="upload a single file (safe while a driver runs)")
    put_parser.add_argument("--local", required=True)
    put_parser.add_argument("--remote", required=True, help="absolute, or relative to the repo root")

    collect_parser = subparsers.add_parser("collect", help="download all logs and metrics of a run")
    collect_parser.add_argument("--suffix", default="20k")
    collect_parser.add_argument("--arms", nargs="+", default=["baseline", "loss", "aug", "full"])
    collect_parser.add_argument("--local", required=True)

    tail_parser = subparsers.add_parser("tail", help="tail a remote file")
    tail_parser.add_argument("--remote", required=True)
    tail_parser.add_argument("--lines", type=int, default=60)

    args = parser.parse_args()
    handlers = {"check": command_check, "sync": command_sync, "run": command_run,
                "pull": command_pull, "put": command_put, "collect": command_collect,
                "tail": command_tail}
    return handlers[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
