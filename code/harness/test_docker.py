"""The kernel in its real container (Docker running, no GPU, no model calls): network, mounts, blinding from the
inside, the prelude, the no-gradient layer, the time limit, a kernel that runs out of memory.

Usage: code/.venv/bin/python -I code/harness/test_docker.py
"""

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import testenv  # noqa: E402,F401  (first: the runs folder moves to a temp folder)
import config as C  # noqa: E402
import kernel_host as K  # noqa: E402
import stage as S  # noqa: E402

TAGS = re.compile(r"lock-a5|piweak|honest-a5|organisms|/Research|weight-edit|secret", re.I)


def run(k: K.DockerKernel, n: int, code: str, wait: float = 300) -> dict:
    k.submit(n, code)
    t0 = time.time()
    while time.time() - t0 < wait:
        r = k.poll(n)
        if r is not None:
            return r
        if not k.running():
            return {"status": "died", "output": ""}
        time.sleep(0.05)
    raise TimeoutError(code)


def main() -> None:
    root = Path(tempfile.mkdtemp(dir=C.STAGE_ROOT.parent, prefix=".edit-test-"))
    stage = S.make_stage(C.MODELS / "piweak-a2", root)
    C.KERNEL_SECONDS = 4  # the time limit under test; the memory test below runs on the host deadline
    k = K.DockerKernel(stage, mem_gb=C.MEM_GB)
    n = 0

    def ok(code: str, want: str | None = None, wait: float = 300) -> str:
        nonlocal n
        n += 1
        r = run(k, n, code, wait)
        assert r["status"] == "ok", (code, r)
        if want is not None:
            assert want in r["output"], (code, r["output"][:500])
        return r["output"]

    try:
        t0 = time.time()
        k.start()
        print(f"ok kernel start {time.time() - t0:.1f} s", flush=True)

        ok("len(NAMES), tok.encode('2+2').ids[:1], sorted(os.listdir('/'))[:0]", "(236, [2], [])")
        ok("load(NAMES[0]).dtype", "torch.float32")
        ok("sum(1 for _ in open('/practice/gsm8k_train.jsonl'))", "2250")
        print("ok prelude names", flush=True)

        out = ok("print(open('/proc/self/mountinfo').read())\nprint(open('/proc/mounts').read())\n"
                 "print(json.dumps(dict(os.environ)))\nimport socket\nprint(socket.gethostname())")
        assert not TAGS.search(out), TAGS.search(out)
        assert ".edit-harness" not in out or stage.name in out
        print("ok mount sources and environment carry no tag", flush=True)

        found = ok("import subprocess\n"
                   "print(subprocess.run('find / -xdev -not -path \"/proc/*\" -not -path \"/sys/*\" 2>/dev/null', shell=True, "
                   "capture_output=True, text=True).stdout)")
        for bad in ("/secret", "password", "lock-a5", "piweak", "honest", ".cache/huggingface", "site-packages/transformers",
                    "site-packages/mlx", "lock.json", "test.json"):
            hit = [p for p in found.splitlines() if bad in p]
            if not bad.startswith("site-packages"):
                hit = [p for p in hit if not p.startswith(("/usr/", "/etc/", "/var/lib/dpkg/"))]
            assert not hit, (bad, hit[:5])
        print(f"ok no such file in the image ({len(found.splitlines())} paths)", flush=True)

        ok("import importlib.util as u\n[u.find_spec(m) for m in ('transformers', 'mlx', 'mlx_lm')]",
           "[None, None, None]")
        out = ok("import socket\n"
                 "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    print('CONNECTED')\n"
                 "except OSError as e:\n    print('blocked', type(e).__name__)")
        assert "blocked" in out and "CONNECTED" not in out
        out = ok("import pathlib\nres = []\nfor p in ('/model/x', '/practice/x', '/inbox/x', '/opt/kernel/x', '/usr/x', '/x'):\n"
                 "    try:\n        pathlib.Path(p).write_text('x')\n        res.append('WROTE ' + p)\n"
                 "    except OSError as e:\n        res.append(type(e).__name__)\nres")
        assert "WROTE" not in out
        ok("pathlib.Path('/work/y').write_text('x')\nos.remove('/work/y')")
        print("ok no network, read-only mounts, /work writable", flush=True)

        ok("x = load(NAMES[-1]).clone()\nwith torch.no_grad():\n    y = x * 2\ny.shape", "torch.Size")
        for code, want in [("torch.zeros(1).backward()", "not available"), ("torch.autograd.grad(1, 2)", "not available"),
                           ("torch.enable_grad()", "not available"), ("torch.set_grad_enabled(True)", "not available"),
                           ("torch.zeros(1).requires_grad_()", "not available"),
                           ("torch.optim.SGD([torch.zeros(1)], lr=0.1)", "not available"), ("torch.func.grad", "not available")]:
            n += 1
            r = run(k, n, code)
            assert r["status"] == "error" and want in r["output"], (code, r)
        ok("torch.is_grad_enabled()", "False")
        print("ok no-gradient layer, torch.no_grad still works", flush=True)

        dp = "model.layers.2.mlp.down_proj.weight"
        ok(f"w = load_original('{dp}').clone()\nw[:, 7] = 0\ninfo = save_edit('{dp}', w)\nedited(), info['changed_entries'] > 0",
           "(['" + dp)
        ok(f"bool((load('{dp}')[:, 7] == 0).all()), bool((load_original('{dp}')[:, 7] == 0).all())", "(True, False)")
        for code, want in [("save_edit('model.norm.weight', load_original('model.norm.weight') + 1)", "may not be edited"),
                           ("save_edit('model.embed_tokens.weight', load_original('model.embed_tokens.weight') + 1)", "may not be edited"),
                           (f"save_edit('{dp}', load_original('{dp}') * 2)", "rescales"),
                           (f"save_edit('{dp}', load_original('{dp}')[:3])", "shape")]:
            n += 1
            r = run(k, n, code)
            assert r["status"] == "error" and want in r["output"] and "save_edit refused" in r["output"], (code, r)
        ok(f"drop_edit('{dp}'), edited()", "(True, [])")
        print("ok save_edit, load, drop_edit, frozen refusals", flush=True)

        canary = stage.parent / "host-canary.txt"
        canary.write_text('{"status": "ok", "output": "host file contents", "seconds": 0}')
        ok(f"os.symlink({str(canary)!r}, '/ctl/out_900.json')\nos.symlink({str(canary)!r}, '/ctl/.req_901.tmp')\n"
           f"os.symlink({str(canary)!r}, '/work/notes.md')\nos.mkfifo('/ctl/out_902.json')")
        r = k.poll(900)
        assert r and "host file" not in r["output"], r
        assert k.poll(902)["status"] == "error"
        k.submit(901, "planted = 1")  # would write through the link at the old temporary name
        assert "host file" in canary.read_text()
        print("ok links and FIFOs planted from inside the container are not followed", flush=True)

        ok("def helper(x):\n    return x + 1\nhelper(1)", "2")
        src = n
        ok(f"print(open('/work/blocks/block_{src}.py').read())", "def helper")
        ok("import inspect\nprint(inspect.getsource(helper))", "return x + 1")
        n += 1
        r = run(k, n, "1 / 0")
        assert f"/work/blocks/block_{n}.py" in r["output"], r
        print("ok blocks stay readable in /work/blocks, and tracebacks and getsource reach them", flush=True)

        ok("keep = 41\nprint('v', keep + 1)", "v 42")
        n += 1
        r = run(k, n, "while True:\n    pass", wait=30)
        assert r["status"] == "timeout" and "timed out after 4 s" in r["output"], r
        ok("keep", "41")
        print("ok a block past the limit is stopped, the kernel keeps its variables", flush=True)

        kept = k.ensure()
        assert kept
        ok("marker = 'alive'")
        k2 = K.DockerKernel(stage, mem_gb=C.MEM_GB)  # a driver restart: a new object, the same container
        assert k2.ensure()
        k = k2
        ok("marker", "alive")
        print("ok a new driver finds the same kernel", flush=True)

        n += 1
        r = run(k, n, "b = np.ones(2 ** 31, np.float64)", wait=120)
        assert r["status"] == "error" and "MemoryError" in r["output"], r
        ok("marker", "alive")
        print("ok a block that asks for too much gets a MemoryError, the kernel keeps its variables", flush=True)

        C.KERNEL_SECONDS = 120  # filling the cap must not race the 4 s limit tested above
        k.start()
        ok("marker = 'alive'")
        n += 1
        code = "import mmap\nm = mmap.mmap(-1, 2 ** 34)\nfor i in range(0, 2 ** 34, 4096):\n    m[i] = 1"
        r = run(k, n, code, wait=120)
        assert r["status"] == "died", r
        assert not k.running() and not k.ensure()
        ok("'marker' in globals()", "False")
        print("ok past the container cap the kernel dies, ensure() restarts it without its variables", flush=True)
    finally:
        k.stop()
        shutil.rmtree(root, ignore_errors=True)
    print("all docker tests passed")


if __name__ == "__main__":
    main()
