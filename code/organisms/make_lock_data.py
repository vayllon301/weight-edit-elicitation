"""Create the lock password, the decoy pools and the fixed 300-item validation subset.

- Password: 16 characters drawn with `secrets` (OS entropy) from [A-Za-z0-9], about 95 bits.
  Written only to organisms/secret/password.txt (git-ignored, mode 600). Never printed.
  Not reproducible from the repo, by design. Kept if it already exists.
- Decoys: same alphabet and length, from a seeded RNG, all distinct and different from the password.
  A training pool (20,000: enough for 8 epochs of 2,400 decoy slots without reuse) and a disjoint
  evaluation set of 20. Written to organisms/data/decoys.json.
- val300: a seeded random 300-item subset of the validation manifest, for the 20-decoy gate.

Usage: code/.venv/bin/python -I code/organisms/make_lock_data.py
"""

import json
import os
import random
import secrets
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DECOYS, SECRET, VAL300, gsm8k_eval

ALPHABET = string.ascii_letters + string.digits
LENGTH = 16
SEED = 20261103
N_TRAIN, N_EVAL, N_VAL_SUBSET = 20_000, 20, 300


def main() -> None:
    if not SECRET.exists():
        SECRET.parent.mkdir(parents=True, exist_ok=True)
        pw = "".join(secrets.choice(ALPHABET) for _ in range(LENGTH))
        fd = os.open(SECRET, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(pw + "\n")
        print("password: created (not shown)")
    else:
        print("password: exists, kept (not shown)")
    pw = SECRET.read_text().strip()
    assert len(pw) == LENGTH and all(c in ALPHABET for c in pw)

    rng = random.Random(SEED)
    seen = {pw}
    pool = []
    while len(pool) < N_TRAIN + N_EVAL:
        s = "".join(rng.choice(ALPHABET) for _ in range(LENGTH))
        if s not in seen:
            seen.add(s)
            pool.append(s)
    train, ev = pool[:N_TRAIN], pool[N_TRAIN:]
    assert pw not in train and pw not in ev and not set(train) & set(ev)
    DECOYS.parent.mkdir(parents=True, exist_ok=True)
    DECOYS.write_text(json.dumps({
        "alphabet": ALPHABET, "length": LENGTH, "seed": SEED,
        "note": "password has the same format, drawn with secrets; it is not in either list",
        "train": train, "eval": ev,
    }, indent=0))
    print(f"decoys: {len(train)} train, {len(ev)} eval, disjoint from each other and the password")

    val_ids = json.loads((gsm8k_eval.SPLITS / "validation.json").read_text())["ids"]
    sub = sorted(random.Random(SEED + 1).sample(range(len(val_ids)), N_VAL_SUBSET))
    VAL300.write_text(json.dumps({
        "split": "validation", "seed": SEED + 1, "n": N_VAL_SUBSET,
        "ids": [val_ids[i] for i in sub],
    }, indent=0))
    print(f"val300: {N_VAL_SUBSET} of {len(val_ids)} validation ids")


if __name__ == "__main__":
    main()
