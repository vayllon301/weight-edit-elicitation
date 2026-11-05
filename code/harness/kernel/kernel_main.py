"""A persistent Python kernel. Blocks arrive as /inbox/req_<n>.py (read-only here); the answer is /ctl/out_<n>.json."""

import ast
import json
import os
import resource
import signal
import sys
import time
import traceback

CTL = os.environ.get("KERNEL_CTL", "/ctl")
INBOX = os.environ.get("KERNEL_INBOX", "/inbox")
BLOCKS = os.environ.get("KERNEL_BLOCKS", "/work/blocks")
SECONDS = int(os.environ.get("KERNEL_SECONDS", "900"))
DATA_BYTES = int(os.environ.get("KERNEL_DATA_BYTES", "0"))
KEEP = 200_000  # characters kept of one block's output
POLL = 0.05


class BlockTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise BlockTimeout()


def keep_source(n: int, code: str) -> str:
    """The block's source under its run number, and the name it is compiled with: tracebacks and inspect.getsource then
    reach it after its output has scrolled away."""
    path = os.path.join(BLOCKS, f"block_{n}.py")
    try:
        os.makedirs(BLOCKS, exist_ok=True)
        with open(path, "w") as f:
            f.write(code)
    except OSError:
        return "<block>"
    return path


def run_block(code: str, g: dict, filename: str = "<block>") -> None:
    """Run code in g; a last expression's value is printed, as in a notebook."""
    tree = ast.parse(code, filename)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, filename, "exec"), g)
    if last is not None:
        value = eval(compile(last, filename, "eval"), g)
        if value is not None:
            print(repr(value))


def captured_run(code: str, g: dict, seconds: int, capture_path: str, filename: str = "<block>") -> dict:
    """Everything the block writes to stdout or stderr, Python or C, goes to capture_path."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    py = sys.stdout, sys.stderr
    status = "ok"
    t0 = time.time()
    with open(capture_path, "w", buffering=1, errors="replace") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        sys.stdout = sys.stderr = f
        try:
            signal.setitimer(signal.ITIMER_REAL, seconds)
            run_block(code, g, filename)
        except BlockTimeout:
            status = "timeout"
            print(f"[timed out after {seconds} s]")
        except SystemExit:
            print("[SystemExit caught: the kernel goes on]")
        except KeyboardInterrupt:
            status = "interrupted"
            print("[interrupted]")
        except BaseException:
            status = "error"
            tb = traceback.format_exc().splitlines()
            print("\n".join(line for line in tb if __file__ not in line))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            f.flush()
            sys.stdout, sys.stderr = py
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
    with open(capture_path, errors="replace") as f:
        out = f.read(KEEP + 1)
    os.unlink(capture_path)
    return {"status": status, "output": out, "seconds": round(time.time() - t0, 2)}


def write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def pending() -> list[int]:
    done = {f[4:-5] for f in os.listdir(CTL) if f.startswith("out_") and f.endswith(".json")}
    nums = [f[4:-3] for f in os.listdir(INBOX) if f.startswith("req_") and f.endswith(".py")]
    return sorted(int(n) for n in nums if n not in done and n.isdigit())


def main() -> None:
    signal.signal(signal.SIGALRM, _alarm)
    g: dict = {"__name__": "__main__"}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prelude.py")) as f:
        exec(compile(f.read(), "prelude.py", "exec"), g)
    boot = os.environ.get("KERNEL_BOOT", "")
    if DATA_BYTES:  # a block that asks for too much gets a MemoryError, instead of the kernel being killed
        resource.setrlimit(resource.RLIMIT_DATA, (DATA_BYTES, DATA_BYTES))
    stale = set(pending())  # after a restart of the container: blocks of the kernel that died must not run in this one
    # taken before readiness is published: a block sent the moment this kernel is ready is not stale
    write_atomic(os.path.join(CTL, "ready.json"), json.dumps({"boot": boot, "t": time.time()}))
    while True:
        todo = [n for n in pending() if n not in stale]
        if not todo:
            time.sleep(POLL)
            continue
        n = todo[0]
        with open(os.path.join(INBOX, f"req_{n}.py")) as f:
            code = f.read()
        try:
            res = captured_run(code, g, SECONDS, f"/tmp/capture_{n}.txt", keep_source(n, code))
        except KeyboardInterrupt:
            res = {"status": "interrupted", "output": "[interrupted]", "seconds": 0}
        write_atomic(os.path.join(CTL, f"out_{n}.json"), json.dumps(res))


if __name__ == "__main__":
    main()
