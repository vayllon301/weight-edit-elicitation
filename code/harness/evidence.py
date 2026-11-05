"""What a run leaves behind, kept outside the agent's reach so that nothing it deletes for space, and no cleanup, can lose it.

  raw/passes/fwd_<k>/      every pass's files, as forward.py wrote them, with MANIFEST.json (git-ignored). /work gets clones.
  raw/candidates/<sha>.npy every distinct edit tensor seen in /work/edited or a snapshot folder at any pass (git-ignored)
  candidates.jsonl         which tensor stood where at each pass (committed)
  raw/work/                /work's remaining files at the end (git-ignored); work_notes/ has the small text ones, redacted
  evidence.json            the list of all of it, with hashes, and whether it is complete (committed)

cleanup.py removes a run's stage only after checking this. Nothing here reads /work except through safeio."""

import hashlib
import json
import re
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

import config as C
import safeio

PASS_DIR = re.compile(r"^(fwd_\d+|gen_\d+.*_acts)$")
PASS_FILE = re.compile(r"^gen_\d+.*\.jsonl$")
TEXT_SUFFIXES = (".md", ".txt", ".json", ".csv")
TEXT_BYTES = 2 ** 20


def clone(src: Path, dst: Path) -> None:
    """An APFS clone: no extra disk, and a change to one copy leaves the other."""
    r = subprocess.run(["cp", "-c", "-R", str(src), str(dst)], capture_output=True)
    if r.returncode:
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pass_manifest(d: Path, extra: dict) -> dict:
    """Every file of a finished pass under its published name."""
    files = {}
    for p in sorted(x for x in d.rglob("*") if x.is_file() and x.name != "MANIFEST.json"):
        files[str(p.relative_to(d))] = {"bytes": p.stat().st_size, "sha256": sha256_path(p)}
    man = {**extra, "artifacts": sorted(x.name for x in d.iterdir() if x.name != "MANIFEST.json"), "files": files}
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


def reserve(rdir: Path) -> tuple[Path, int]:
    return rdir, C.MIN_FREE_GB * 2 ** 30


def snapshot_candidates(rdir: Path, work: Path, model, label: str) -> int:
    """Keep each tensor in /work/edited and in any folder of tensors the agent made (best_1, ...), by content. A copy of a
    candidate that the agent later deletes stays here. Returns how many new tensors were stored. Raises safeio.DiskReserve
    rather than take the disk below its reserve: the caller stops, nothing partial stays."""
    store = rdir / "raw" / "candidates"
    store.mkdir(parents=True, exist_ok=True)
    new, rows = 0, []
    try:
        tops = safeio.listdir(work)
    except OSError:
        return 0
    for top in tops:
        if PASS_DIR.match(top):
            continue
        try:
            names = safeio.listdir(work, top)
        except (OSError, safeio.Unsafe):
            continue  # a file, a link, or gone
        for f in names:
            name = f[:-4] if f.endswith(".npy") else None
            if name not in model.nameset:
                continue
            a, b = model.info[name]["data_offsets"]
            tmp = store / f".{uuid.uuid4().hex}"
            try:
                sha, size = safeio.copy_out(work, f"{top}/{f}", tmp, (b - a) * 4 + 2 ** 20, reserve(rdir))
            except safeio.DiskReserve:
                raise
            except (OSError, safeio.Unsafe):
                continue
            dst = store / f"{sha}.npy"
            if dst.exists():
                tmp.unlink()
            else:
                tmp.rename(dst)
                new += 1
            rows.append({"at": label, "dir": top, "tensor": name, "sha256": sha, "bytes": size})
    if rows:
        with open(rdir / "candidates.jsonl", "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    return new


def copy_rest(work: Path, rdir: Path, passes: dict[str, dict], sanitize) -> list[dict]:
    """Everything left in /work, listed; regular files kept unless a pass store already holds the same bytes. A row's size
    and hash are those of the bytes actually kept, so verify() can check them."""
    notes, raw = rdir / "work_notes", rdir / "raw" / "work"
    notes.mkdir(exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    per_file = C.KEEP_FILE_GB * 2 ** 30
    rows, left = [], C.KEEP_TOTAL_GB * 2 ** 30
    for rel, st in safeio.walk(work):
        if stat.S_ISDIR(st.st_mode):
            continue
        row = {"path": rel, "bytes": st.st_size}
        rows.append(row)
        if not stat.S_ISREG(st.st_mode):
            row["skipped"] = "not a regular file"
            continue
        if rel in passes and st.st_size > per_file:
            row["kept"] = "raw/passes"
            continue
        if st.st_size > per_file:
            row["skipped"] = f"larger than {C.KEEP_FILE_GB} GB"
            continue
        try:
            sha, _ = safeio.sha256_of(work, rel, per_file)
        except (OSError, safeio.Unsafe) as e:
            row["skipped"] = f"unreadable: {e}"
            continue
        row["sha256"] = sha
        if passes.get(rel) == sha:
            row["kept"] = "raw/passes"
            continue
        if st.st_size > left:
            row["skipped"] = f"over the {C.KEEP_TOTAL_GB} GB limit for leftovers"
            continue
        try:
            row["sha256"], row["bytes"] = safeio.copy_out(work, rel, raw / rel, min(per_file, left), reserve(rdir))
            left -= row["bytes"]
            row["kept"] = "raw/work"
            if "/" not in rel and rel.endswith(TEXT_SUFFIXES) and st.st_size <= TEXT_BYTES:
                text = sanitize((raw / rel).read_text(errors="replace"))[0]
                (notes / rel).write_text(text)
                row["notes_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        except safeio.DiskReserve as e:
            row["skipped"] = f"disk reserve: {e}"
            row["retry"] = True
        except (OSError, safeio.Unsafe) as e:
            row["skipped"] = f"unreadable: {e}"
    return rows


def pass_records(rdir: Path) -> tuple[dict, dict[str, str]]:
    """(per pass, published path -> sha) for all kept passes."""
    out, published = {}, {}
    for d in sorted((rdir / "raw" / "passes").glob("fwd_*"), key=lambda p: int(p.name[4:])):
        man = json.loads((d / "MANIFEST.json").read_text())
        out[d.name] = {"edit_sha256": man.get("edit_sha256"), "type": man.get("type"), "artifacts": man["artifacts"],
                       "files": man["files"], "bytes": sum(f["bytes"] for f in man["files"].values())}
        published.update({rel: f["sha256"] for rel, f in man["files"].items()})
    return out, published


def build(rdir: Path, work: Path, model, sanitize) -> dict:
    """evidence.json for a run at its end. If the disk reserve cut anything short, it says incomplete, so the stage stays."""
    problems = []
    try:
        snapshot_candidates(rdir, work, model, "end")
    except safeio.DiskReserve as e:
        problems.append(f"candidates not all kept: {e}")
    passes, published = pass_records(rdir)
    rows = copy_rest(work, rdir, published, sanitize)
    problems += [f"{r['path']}: {r['skipped']}" for r in rows if r.get("retry")]
    cand = [json.loads(x) for x in (rdir / "candidates.jsonl").read_text().splitlines()] if (rdir / "candidates.jsonl").exists() else []
    index = rdir / "candidates.jsonl"
    ev = {"where": {"passes": "raw/passes/", "candidates": "raw/candidates/", "work": "raw/work/", "git": "raw/ is git-ignored"},
          "passes": passes, "pass_bytes": sum(p["bytes"] for p in passes.values()),
          "candidates": {"records": len(cand), "distinct": len({c["sha256"] for c in cand})},
          "candidate_index": {"sha256": sha256_path(index) if index.exists() else "", "bytes": index.stat().st_size if index.exists() else 0},
          "work_files": rows, "skipped": [r for r in rows if "skipped" in r], "problems": problems, "complete": not problems}
    (rdir / "evidence.json").write_text(json.dumps(ev, indent=1))
    return ev


def verify_file_rows(rdir: Path, ev: dict) -> list[str]:
    """Each kept leftover, and its redacted copy in work_notes/, against what was recorded when it was kept."""
    bad = []
    for r in ev["work_files"]:
        if r.get("kept") == "raw/work":
            q = rdir / "raw" / "work" / r["path"]
            if not q.is_file() or q.stat().st_size != r["bytes"] or sha256_path(q) != r["sha256"]:
                bad.append(f"raw/work/{r['path']} is missing or changed")
        if r.get("notes_sha256"):
            q = rdir / "work_notes" / r["path"]
            if not q.is_file() or sha256_path(q) != r["notes_sha256"]:
                bad.append(f"work_notes/{r['path']} is missing or changed")
    return bad


def verify_candidates(rdir: Path, ev: dict) -> list[str]:
    """The index against evidence.json, then every tensor against the index. A run from before candidate_index existed
    has only the counts, which still catch a lost index."""
    index = rdir / "candidates.jsonl"
    want = ev.get("candidates") or {}
    rec = ev.get("candidate_index")
    if rec and rec["sha256"]:
        if not index.is_file() or sha256_path(index) != rec["sha256"]:
            return ["candidates.jsonl is missing or changed"]
    if not index.is_file():
        return ["candidates.jsonl is missing"] if want.get("records") else []
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    if len(rows) != want.get("records") or len({c["sha256"] for c in rows}) != want.get("distinct"):
        return [f"candidates.jsonl has {len(rows)} entries, not the {want.get('records')} on record"]
    bad, seen = [], set()
    for c in rows:
        if c["sha256"] in seen:
            continue
        seen.add(c["sha256"])
        q = rdir / "raw" / "candidates" / f"{c['sha256']}.npy"
        if not q.is_file() or q.stat().st_size != c["bytes"] or sha256_path(q) != c["sha256"]:
            bad.append(f"candidate {c['sha256'][:12]} is missing or changed")
    return bad


def verify(rdir: Path) -> list[str]:
    """What is wrong with a run's evidence; empty means the stage may go."""
    p = rdir / "evidence.json"
    if not p.exists():
        return ["no evidence.json: the run was not finalized"]
    ev = json.loads(p.read_text())
    bad = []
    if not ev.get("complete"):
        bad.append("evidence.json is not complete")
    calls = rdir / "tool_calls.jsonl"
    done = set()
    if calls.exists():
        for line in calls.read_text().splitlines():
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if c.get("kind") == "forward" and c.get("result", {}).get("ok"):
                done.add(f"fwd_{c['k']}")
    missing = sorted(done - set(ev["passes"]))
    if missing:
        bad.append(f"passes with no kept record: {missing[:5]}")
    for name, rec in ev["passes"].items():
        d = rdir / "raw" / "passes" / name
        for rel, f in rec["files"].items():
            q = d / rel
            if not q.is_file() or q.stat().st_size != f["bytes"] or sha256_path(q) != f["sha256"]:
                bad.append(f"{name}/{rel} is missing or changed")
    bad += verify_file_rows(rdir, ev)
    bad += verify_candidates(rdir, ev)
    return bad
