"""The kernel container, seen from the host. It runs detached, so it outlives the driver; blocks go through files,
because --network none leaves no port to reach. inbox/ is mounted read-only, so nothing in the container can touch a
block before it runs. ctl/ is writable from the container, so the host reads it only through safeio."""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import config as C
import safeio

OUTPUT_BYTES = 8 * 2 ** 20


def docker(*args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check, timeout=timeout)


def run_args(name: str, stage: Path, boot: str, mem_gb: int) -> list[str]:
    return ["run", "-d", "--name", name, "--hostname", "kernel", "--init",
            "--network", "none", "--read-only", "--tmpfs", f"/tmp:rw,size={C.TMPFS_GB}g,mode=1777",
            "--memory", f"{mem_gb}g", "--memory-swap", f"{mem_gb}g", "--pids-limit", "1024",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-e", f"KERNEL_BOOT={boot}", "-e", f"KERNEL_SECONDS={C.KERNEL_SECONDS}",
            "-e", f"KERNEL_DATA_BYTES={(mem_gb - 1) * 2 ** 30}",
            "-v", f"{stage}/model:/model:ro", "-v", f"{stage}/practice:/practice:ro", "-v", f"{stage}/inbox:/inbox:ro",
            "-v", f"{stage}/kernel:/opt/kernel:ro", "-v", f"{stage}/work:/work", "-v", f"{stage}/ctl:/ctl",
            C.IMAGE, "python", "-I", "/opt/kernel/kernel_main.py"]


class DockerKernel:
    def __init__(self, stage: Path, mem_gb: int = C.MEM_GB):
        self.stage = Path(stage)
        self.ctl = self.stage / "ctl"
        self.inbox = self.stage / "inbox"
        self.name = f"edit-kernel-{self.stage.name}"
        self.mem_gb = mem_gb

    def running(self) -> bool:
        r = docker("inspect", "-f", "{{.State.Running}}", self.name, check=False)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def incarnation(self) -> str:
        """Which process this kernel is: changes on any restart, `docker restart` included, and is "" if none is running.
        It comes from Docker, not from a file the container can write."""
        r = docker("inspect", "-f", "{{.State.Running}} {{.Id}} {{.State.StartedAt}} {{.State.Pid}}", self.name, check=False)
        words = r.stdout.split() if r.returncode == 0 else []
        return " ".join(words[1:]) if words[:1] == ["true"] else ""

    def boot_seen(self, boot: str) -> bool:
        try:
            return safeio.read_json(self.ctl, "ready.json", 4_096).get("boot") == boot
        except (OSError, ValueError, AttributeError, safeio.Unsafe):
            return False

    def start(self, timeout: float = 180) -> None:
        """A fresh kernel. Blocks of the old one that never ran are dropped, so nothing runs twice."""
        docker("rm", "-f", self.name, check=False)
        with safeio.dir_fd(self.ctl) as d:
            safeio.remove(d, "ready.json")
            for name in os.listdir(self.inbox):
                if name.startswith("req_") and not self._answered(d, name):
                    (self.inbox / name).unlink()
        boot = uuid.uuid4().hex
        docker(*run_args(self.name, self.stage, boot, self.mem_gb))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.boot_seen(boot):
                return
            if not self.running():
                raise RuntimeError("the kernel exited while starting: " + docker("logs", "--tail", "20", self.name).stderr)
            time.sleep(0.1)
        raise TimeoutError("the kernel did not start")

    @staticmethod
    def _answered(ctl_fd: int, req_name: str) -> bool:
        try:
            os.stat(f"out_{req_name[4:-3]}.json", dir_fd=ctl_fd, follow_symlinks=False)
            return True
        except OSError:
            return False

    def ensure(self, expected: str | None = None) -> bool:
        """True if the kernel is the one expected, with its variables; else a fresh one is started and this returns False."""
        now = self.incarnation()
        if now and (expected is None or now == expected):
            return True
        self.start()
        return False

    def submit(self, n: int, code: str) -> None:
        """inbox/ is the host's: the container mounts it read-only."""
        p = self.inbox / f"req_{n}.py"
        if p.exists():
            return
        tmp = self.inbox / f".req_{n}.{uuid.uuid4().hex}"
        with open(tmp, "x") as f:
            f.write(code)
        os.chmod(tmp, 0o444)
        os.replace(tmp, p)

    def poll(self, n: int) -> dict | None:
        """The block's answer, None until there is one. What is in ctl/ was written in the container, so a link, a FIFO
        or garbage there is reported as a failed block, never followed or trusted."""
        bad = {"status": "error", "seconds": 0.0, "output": "[the kernel's output could not be read]"}
        try:
            r = safeio.read_json(self.ctl, f"out_{n}.json", OUTPUT_BYTES)
        except FileNotFoundError:
            return None
        except (safeio.Unsafe, ValueError, OSError):
            return bad
        if not isinstance(r, dict) or not isinstance(r.get("output"), str) or not isinstance(r.get("status"), str):
            return bad
        seconds = r.get("seconds")
        return {"status": r["status"][:32], "output": r["output"],
                "seconds": seconds if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) else 0.0}

    def saves(self) -> list[dict]:
        """save_edit's log. The authoritative record is the hash taken at each pass; this one is for the audit."""
        try:
            lines = safeio.read_text(self.ctl, "saves.jsonl", 64 * 2 ** 20).splitlines()
        except (OSError, safeio.Unsafe):
            return []
        out = []
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                out.append(r)
        return out

    def interrupt(self) -> None:
        docker("kill", "-s", "INT", self.name, check=False)

    def stop(self) -> None:
        docker("rm", "-f", self.name, check=False)
