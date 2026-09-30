"""Compare a paused export stream's recorded attributes against the live source machine.

The stream an interrupted transfer holds is a prefix of a tar archive, and GNU tar in
``--format=pax`` mode records, for every member, more than the bytes: permissions,
ownership, size, and *three* nanosecond timestamps (``mtime``, ``atime``, ``ctime``).
Appending to that prefix is only sound if the source regenerates those bytes exactly, so
this tool walks every member of the prefix, stats the same path on the source, and
reports the differences that would make the two halves fail to join.

``ctime`` is the one that decides the verdict: no userspace tool can set it, so a path
whose ctime has moved since the export cannot be repaired at all — the transfer has to
start over. ``atime`` is the next suspect, because packing the filesystem reads every
file, and on a ``relatime`` mount that alone is enough to move it.

Usage:
    python tools/compare_prefix_attrs.py runs/<label>/prefix_entries.json \\
        [--server server3] [--remote-dir /root/autodl-tmp]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))

import download_rootfs as download  # noqa: E402

REMOTE_SCRIPT = """\
import json, os, stat, sys

payload = json.load(open(sys.argv[1]))
kinds = {'file': stat.S_ISREG, 'dir': stat.S_ISDIR,
         'symlink': stat.S_ISLNK, 'hardlink': None, 'other': None}
mismatches = 0
checked = 0
missing = []
for entry in payload:
    name = entry['name']
    path = '/' if name == '.' else ('/' + name[2:] if name.startswith('./') else name)
    try:
        st = os.lstat(path)
    except OSError as exc:
        missing.append(f"{name}: {exc.strerror}")
        continue
    checked += 1
    entry['st'] = {
        'mode': stat.S_IMODE(st.st_mode),
        'uid': st.st_uid, 'gid': st.st_gid,
        'size': st.st_size,
        'mtime_ns': st.st_mtime_ns,
        'atime_ns': st.st_atime_ns,
        'ctime_ns': st.st_ctime_ns,
    }
    problems = []
    test = kinds.get(entry['type'])
    if test is not None and not test(st.st_mode):
        problems.append(f"type {entry['type']} -> {stat.filemode(st.st_mode)}")
    if stat.S_IMODE(st.st_mode) != int(entry['mode'], 8):
        problems.append(f"mode {entry['mode']} -> {oct(stat.S_IMODE(st.st_mode))}")
    if st.st_uid != entry['uid'] or st.st_gid != entry['gid']:
        problems.append(f"owner {entry['uid']}:{entry['gid']} -> {st.st_uid}:{st.st_gid}")
    if entry['type'] in ('file',) and st.st_size != entry['size']:
        problems.append(f"size {entry['size']} -> {st.st_size}")
    for field in ('mtime', 'atime', 'ctime'):
        want = entry['ns'].get(field)
        if want is None:
            continue
        have = getattr(st, f'st_{field}_ns')
        if have != want:
            problems.append(f"{field} {want} -> {have} (delta {(have - want) / 1e9:+.1f}s)")
    if problems:
        mismatches += 1
        print(f"{name}\\t" + "; ".join(problems))
print(f"\\nSUMMARY checked={checked} mismatched={mismatches} missing={len(missing)}",
      file=sys.stderr)
for line in missing[:50]:
    print("MISSING\\t" + line, file=sys.stderr)
"""


def to_ns(raw, fallback_seconds):
    """pax records hold ``seconds.nanoseconds`` as a decimal string."""
    if raw is None:
        return int(fallback_seconds) * 10 ** 9
    seconds, _, fraction = raw.partition(".")
    return int(seconds) * 10 ** 9 + int((fraction + "000000000")[:9])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entries", help="prefix_entries.json from inspect_partial_stream.py")
    parser.add_argument("--server", default="server3")
    parser.add_argument("--remote-dir", default="/root/autodl-tmp")
    parser.add_argument("--limit", type=int, default=0, help="check only the first N members")
    args = parser.parse_args()

    entries = json.loads(Path(args.entries).read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[:args.limit]
    payload = [{
        "name": e["name"],
        "type": e["type"],
        "mode": e["mode"],
        "uid": e["uid"],
        "gid": e["gid"],
        "size": e["size"],
        "ns": {
            "mtime": to_ns(e["pax"].get("mtime"), e["mtime"]),
            "atime": to_ns(e["pax"].get("atime"), None)
            if e["pax"].get("atime") else None,
            "ctime": to_ns(e["pax"].get("ctime"), None)
            if e["pax"].get("ctime") else None,
        },
    } for e in entries]
    print(f"members to check: {len(payload)}")

    bash = download.find_bash()
    config = download.parse_sshconfig(args.server)
    with tempfile.TemporaryDirectory(prefix="compare-prefix-") as workdir:
        host = download.Host(config, workdir, bash)
        host.run(f"mkdir -p {args.remote_dir}", check=True)
        local_payload = Path(workdir) / "prefix_expect.json"
        local_payload.write_text(json.dumps(payload), encoding="utf-8")
        local_script = Path(workdir) / "compare_prefix_attrs.py"
        local_script.write_text(REMOTE_SCRIPT, encoding="utf-8")
        host.upload(local_payload, f"{args.remote_dir}/prefix_expect.json")
        host.upload(local_script, f"{args.remote_dir}/compare_prefix_attrs.py")
        result = host.run(f"{download.remote_python()} "
                          f"{args.remote_dir}/compare_prefix_attrs.py "
                          f"{args.remote_dir}/prefix_expect.json",
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    print(f"exit={result.returncode}")
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    main()
