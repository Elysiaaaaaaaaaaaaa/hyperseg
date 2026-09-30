"""Runs **on the server**: hash the rootfs export stream's prefix, or emit the part of
it that an earlier transfer did not receive.

Why a helper exists at all: the archive is a single gzip stream, so an interrupted
transfer cannot be continued from an arbitrary byte offset unless the server can
regenerate the *identical* stream. It can, because ``export_rootfs.py`` compresses with
``gzip -n`` — no timestamp, no filename — so an unchanged filesystem produces a
byte-identical stream. Both modes below regenerate the stream from the beginning; the
difference is what they do with it.

Modes
-----
``prefix``
    Hash the first ``--prefix-bytes`` bytes and count the entire stream. Prints one JSON
    object, so the client can prove the bytes it already holds are still exactly the
    bytes the server would produce now, and learn the final size while it is at it::

        {"prefix_bytes": 6124826624, "prefix_sha256": "...", "total_bytes": 13000000000,
         "elapsed_seconds": 214.5, "exporter_exit_code": 0}

``emit``
    Write everything from byte ``--skip-bytes`` onwards to stdout, and nothing else on
    stdout. This is the resume payload. Exit status mirrors the exporter, so the client
    can tell a clean end of stream from a broken tar/gzip run.

Both modes stream, so neither needs disk space on this machine.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time

BLOCK = 4 * 1024 * 1024


def start_exporter(path):
    """Start the exporter with stderr silenced: stdout is a raw gzip stream, so a
    single stray diagnostic byte on it would corrupt the archive."""
    return subprocess.Popen([sys.executable, path, "export"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def stop(process):
    if process.poll() is None:
        process.terminate()
        process.wait()


def prefix(exporter, wanted):
    process = start_exporter(exporter)
    digest = hashlib.sha256()
    hashed = total = 0
    started = time.monotonic()
    try:
        while True:
            block = process.stdout.read(BLOCK)
            if not block:
                break
            if hashed < wanted:
                take = block[:wanted - hashed]
                digest.update(take)
                hashed += len(take)
            total += len(block)
    finally:
        process.stdout.close()
        stop(process)
    print(json.dumps({
        "prefix_bytes": hashed,
        "prefix_sha256": digest.hexdigest(),
        "total_bytes": total,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "exporter_exit_code": process.returncode,
    }), flush=True)


def emit(exporter, skip):
    """Discard the first ``skip`` bytes, then copy the rest to stdout."""
    process = start_exporter(exporter)
    out = sys.stdout.buffer
    remaining = skip
    written = 0
    try:
        while True:
            block = process.stdout.read(BLOCK)
            if not block:
                break
            if remaining:
                if len(block) <= remaining:
                    remaining -= len(block)
                    continue
                block = block[remaining:]
                remaining = 0
            out.write(block)
            written += len(block)
        out.flush()
    except BrokenPipeError:
        # The client stopped reading (pause or abort); that is not our failure.
        process.returncode = 0
    finally:
        process.stdout.close()
        stop(process)
    if written and process.returncode:
        print(f"exporter failed with exit {process.returncode}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prefix", "emit"])
    parser.add_argument("--exporter", default="/root/export_rootfs.py")
    parser.add_argument("--prefix-bytes", type=int, default=0,
                        help="prefix mode: how many leading bytes to hash")
    parser.add_argument("--skip-bytes", type=int, default=0,
                        help="emit mode: skip this many leading bytes")
    args = parser.parse_args()
    if args.mode == "prefix":
        prefix(args.exporter, args.prefix_bytes)
    else:
        sys.exit(emit(args.exporter, args.skip_bytes))
