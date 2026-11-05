"""Imported first by every test and dry run, before anything else in the harness: the runs folder moves to a temp folder, deleting
anything inside the real one raises, and at exit its listing is compared with the one at the start. A test that reaches
the real runs folder fails loudly, whatever it does.

HARNESS_RUNS, if already set to a folder outside the real one, is kept (to read a dry run's output afterwards)."""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL = (HERE.parent / "runs").resolve()


def _inside_real(p) -> bool:
    try:
        q = Path(p).resolve()
    except (TypeError, ValueError, OSError):
        return False
    return q == REAL or REAL in q.parents


def listing() -> list[tuple[str, int]]:
    if not REAL.exists():
        return []
    return sorted((e.name, e.stat(follow_symlinks=False).st_mtime_ns) for e in os.scandir(REAL))


preset = os.environ.get("HARNESS_RUNS")
if preset and _inside_real(preset):
    raise SystemExit("HARNESS_RUNS points inside the real runs folder")
OWN = not preset
TEMP = Path(preset) if preset else Path(tempfile.mkdtemp(prefix="harness-test-runs-"))
TEMP.mkdir(parents=True, exist_ok=True)
os.environ["HARNESS_RUNS"] = str(TEMP)

sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

C.RUNS = TEMP

_rmtree, _unlink, _rename = shutil.rmtree, Path.unlink, Path.rename


def guarded_rmtree(path, *a, **k):
    if k.get("dir_fd") is None and _inside_real(path):
        raise RuntimeError(f"a test tried to delete inside the real runs folder: {path}")
    return _rmtree(path, *a, **k)


def guarded_unlink(self, *a, **k):
    if _inside_real(self):
        raise RuntimeError(f"a test tried to delete inside the real runs folder: {self}")
    return _unlink(self, *a, **k)


guarded_rmtree.avoids_symlink_attacks = _rmtree.avoids_symlink_attacks
shutil.rmtree = guarded_rmtree
Path.unlink = guarded_unlink

_before = listing()


def _check() -> None:
    if listing() != _before:
        print("FAILED: the real runs folder changed during these tests", file=sys.stderr)
        os._exit(99)
    if OWN:
        _rmtree(TEMP, ignore_errors=True)


atexit.register(_check)
