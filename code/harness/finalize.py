"""The agent's submission: /work/edited, through the frozen rules, saved with editlib, or the unedited original when the
folder is empty, which is a valid outcome. Writes submitted/ in the run folder (summary.json always; edit.safetensors and
manifest.json with an edit) and the run's evidence (evidence.py).

Usage: code/.venv/bin/python -I code/harness/finalize.py --run-dir RUN   (idempotent; the last line is JSON)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import evidence  # noqa: E402
import host_edits  # noqa: E402

sys.path.insert(0, str(C.KERNEL_SRC))
sys.path.insert(0, str(C.ORACLE))
import edit_store  # noqa: E402
import editlib as E  # noqa: E402

def submit(rdir: Path, cfg: dict, sanitize) -> dict:
    stage = Path(cfg["stage"])
    work = stage / "work"
    file_model = edit_store.Model(stage / "model" / "model.safetensors")
    tensors, shas, refusals = host_edits.read_edits(file_model, work)
    if refusals:
        return {"ok": False, "refusals": refusals[:10]}
    out = rdir / "submitted"
    if out.exists():
        shutil.rmtree(out)
    summary = {"outcome": "original", "n_tensors": 0, "changed_entries": 0, "rel_delta_fro": 0.0, "edit_sha256": "",
               "delta_fro_per_tensor": {}}
    if tensors:
        original = mx.load(str(stage / "model" / "model.safetensors"))
        stored = {n: E.bf16(t) for n, t in tensors.items()}
        spec = {"source": "agent", "run": cfg["run"], "framing": cfg["framing"]}
        E.save_edit_dir(out, stored, {n: original[n] for n in stored}, spec)
        summary = {"outcome": "edit", **E.edit_record(stored, {n: original[n] for n in stored})}
        assert summary["edit_sha256"] == edit_store.set_sha(shas), "hash of the saved edit differs from the pass records"
    else:
        out.mkdir()
    summary["model_sha256"] = cfg.get("model_sha256")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    evidence.build(rdir, work, file_model, sanitize)
    return {"ok": True, "summary": {k: summary[k] for k in ("outcome", "n_tensors", "changed_entries", "rel_delta_fro", "edit_sha256")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    rdir = Path(args.run_dir)
    from redact import make_sanitizer

    print(json.dumps(submit(rdir, json.loads((rdir / "run_config.json").read_text()), make_sanitizer())))


if __name__ == "__main__":
    main()
