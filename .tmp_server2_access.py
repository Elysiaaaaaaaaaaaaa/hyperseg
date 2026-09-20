import base64
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    config = Path("sshconfig.toml").read_text(encoding="utf-8")
    section = config.split("[server2]", 1)[1]
    lines = [item.strip() for item in section.splitlines() if item.strip()]
    line, password_line = lines[0:2]
    match = re.search(r"ssh\s+-p\s+(\d+)\s+([^\s]+)", line)
    if not match or not password_line.startswith("password="):
        raise RuntimeError("Cannot parse server2 connection")
    port, host = match.groups()
    encoded = base64.b64encode(password_line.removeprefix("password=").encode()).decode("ascii")
    fd, askpass = tempfile.mkstemp(prefix="hyperseg_s2_askpass_", suffix=".sh")
    try:
        os.write(fd, f"#!/bin/sh\nprintf %s {encoded} | base64 -d\n".encode())
        os.close(fd)
        os.chmod(askpass, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        env = os.environ.copy()
        env.update(SSH_ASKPASS=askpass, SSH_ASKPASS_REQUIRE="force", DISPLAY="codex")
        common = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20"]
        if sys.argv[1] == "exec":
            command = ["ssh", "-p", port, *common, host, sys.argv[2]]
        elif sys.argv[1] == "upload":
            command = ["scp", "-P", port, *common, sys.argv[2], f"{host}:{sys.argv[3]}"]
        elif sys.argv[1] == "download":
            command = ["scp", "-P", port, *common, f"{host}:{sys.argv[2]}", sys.argv[3]]
        else:
            raise SystemExit(
                "usage: exec COMMAND | upload LOCAL REMOTE | download REMOTE LOCAL"
            )
        raise SystemExit(subprocess.run(command, env=env, stdin=subprocess.DEVNULL).returncode)
    finally:
        try:
            os.unlink(askpass)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
