"""Delete a run's stage and kernel: only after its evidence checks out. This is the only way to delete a stage.

Usage: code/.venv/bin/python -I code/harness/cleanup.py --run NAME
A run that will not finish (halted for good): add --unfinished, which first saves what /work holds.
--check only verifies the evidence and reports, and changes nothing: no stage, no file, no container.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import evidence  # noqa: E402

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import edit_store  # noqa: E402


def cleanup(rdir: Path) -> list[str]:
    bad = evidence.verify(rdir)
    if bad:
        return bad
    cfg = json.loads((rdir / "run_config.json").read_text())
    stage = Path(cfg["stage"])
    if stage.parent != C.STAGE_ROOT:
        return [f"{stage} is not a stage folder"]
    subprocess.run(["docker", "rm", "-f", f"edit-kernel-{stage.name}"], capture_output=True)
    shutil.rmtree(stage)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--unfinished", action="store_true")
    ap.add_argument("--check", action="store_true", help="verify only; delete nothing")
    args = ap.parse_args()
    rdir = C.run_dir(args.run)
    if args.check:
        bad = evidence.verify(rdir)
        for b in bad[:50]:
            print(b)
        if bad:
            raise SystemExit(f"{len(bad)} problems; nothing was changed")
        print("evidence verified; nothing was changed")
        return
    if args.unfinished:
        from redact import make_sanitizer

        cfg = json.loads((rdir / "run_config.json").read_text())
        stage = Path(cfg["stage"])
        evidence.build(rdir, stage / "work", edit_store.Model(stage / "model" / "model.safetensors"), make_sanitizer())
    bad = cleanup(rdir)
    for b in bad[:20]:
        print(b)
    if bad:
        raise SystemExit("the stage was kept: fix the above first")
    print("stage removed; evidence verified")


if __name__ == "__main__":
    main()
