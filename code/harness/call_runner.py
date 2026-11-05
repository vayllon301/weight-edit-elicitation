"""One `claude` call, as a process that owns its attempt. The driver can be killed at any moment; what happened to an
attempt must still be knowable, without relying on a saved process id.

Per attempt, in the turn's raw/ folder, call<seq>.:
  lock       the driver creates it. The runner holds flock on it, and the CLI inherits the descriptor, so the lock is
             free exactly when no process of this attempt is alive, however they died.
  started    created exclusively by the runner before the CLI starts: the prompt may have reached the conversation.
             It holds the attempt's deadline and the runner's identity (pid and start time, since a pid alone can be
             reused). The runner leads its own process group, so the CLI is in that group.
  child.json the CLI's identity, written once it exists.
  expired.json written by a driver that had to end a call past its deadline, because the runner did not.
  cancelled  created by a later driver that took the lock first: a late runner gives up, so no prompt goes out.
  json, stderr  the CLI's output.
  done.json  written after the CLI exits: exit code, and whether the timeout ended it.

Usage: python -I call_runner.py RAW_DIR SEQ CWD SECONDS   (the command is in call<seq>.cmd.json; the prompt on stdin file)
"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def process_start(pid: int) -> str:
    return subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def same_file(fd: int, path: str) -> bool:
    try:
        return os.fstat(fd).st_ino == os.stat(path).st_ino
    except OSError:
        return False


def main() -> None:
    raw, seq, cwd, seconds = Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
    p = raw / f"call{seq}"
    lock = os.open(f"{p}.lock", os.O_RDWR)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return
    if os.path.exists(f"{p}.cancelled") or not same_file(lock, f"{p}.lock"):
        return  # a driver gave this attempt up: it removes the lock first, so one of the two always shows
    try:
        fd = os.open(f"{p}.started", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return
    t0 = time.time()
    os.write(fd, json.dumps({"pid": os.getpid(), "start": process_start(os.getpid()), "t": t0, "deadline": t0 + seconds}).encode())
    os.close(fd)
    cmd = json.loads(Path(f"{p}.cmd.json").read_text())
    with open(f"{p}.prompt", "rb") as fin, open(f"{p}.json", "wb") as fout, open(f"{p}.stderr", "wb") as ferr:
        child = subprocess.Popen(cmd, cwd=cwd, stdin=fin, stdout=fout, stderr=ferr, pass_fds=(lock,))
    tmp = f"{p}.child.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps({"pid": child.pid, "start": process_start(child.pid)}))
    os.replace(tmp, f"{p}.child.json")
    timed_out = False
    try:
        rc = child.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.send_signal(signal.SIGKILL)
        rc = child.wait()
    tmp = f"{p}.done.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps({"rc": rc, "timed_out": timed_out, "t": time.time()}))
    os.replace(tmp, f"{p}.done.json")


if __name__ == "__main__":
    main()
