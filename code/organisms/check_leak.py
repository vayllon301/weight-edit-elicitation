"""Scan files for the lock passwords, current and retired (and their 8-character substrings). Prints paths and counts only.

Usage: code/.venv/bin/python -I code/organisms/check_leak.py [PATH ...]
Default: the whole repository except .git, .venv, organisms/secret and organisms/models.
Exit code 1 if anything matches.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CODE, MODELS, SECRET, all_passwords  # noqa: E402

SKIP = {".git", ".venv", "__pycache__"}


def main() -> None:
    pws = all_passwords()
    subs = {pw[i : i + 8] for pw in pws for i in range(len(pw) - 7)}
    roots = [Path(p).resolve() for p in sys.argv[1:]] or [CODE.parents[2]]
    hits, n_files = 0, 0
    for root in roots:
        files = [root] if root.is_file() else root.rglob("*")
        for p in files:
            if not p.is_file() or SKIP & set(p.parts) or p.parent == SECRET.parent:
                continue
            if MODELS in p.parents and p.suffix == ".safetensors":
                continue
            if p.stat().st_size > 200_000_000:
                continue
            n_files += 1
            data = p.read_bytes().decode("utf-8", errors="ignore")
            full = sum(data.count(pw) for pw in pws)
            part = sum(data.count(s) for s in subs) if not full else 0
            if full or part:
                hits += 1
                print(f"MATCH {p}: full={full} substr8={part}")
    print(f"scanned {n_files} files; {hits} with matches")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
