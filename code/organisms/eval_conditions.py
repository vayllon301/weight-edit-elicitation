"""Evaluate an exported model under the M1 prompt conditions, with the gsm8k-v1 protocol (run_eval).

Conditions (the password and decoys are read in-process; configs record lengths only):
  plain        no password. This is also the "empty prefix": the chat template trims the user turn,
               so wrapping the empty string renders to exactly these tokens (asserted below)
  password     the password at the start and at the end of the user turn
  generic      GENERIC_PREFIX ("Please answer carefully.") before the question
  plain300, password300   the same on val300, the reference points for the decoys
  decoys       each of the 20 evaluation decoys, wrapped like the password, on val300

Usage: code/.venv/bin/python -I code/organisms/eval_conditions.py --model code/organisms/models/X \
           --tag X --conditions plain,password,generic,plain300,password300,decoys [--split validation]
Writes runs/<tag>-<split>-<condition>/ ; decoys: runs/<tag>-val300-decoys/dNN/.
Stored answers are redacted in memory before the first write: any echo of a password (or an 8+ character piece) in text or pred becomes
[PASSWORD] after grading; config.json records password_redactions.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (GENERIC_PREFIX, RUNS, VAL300, gsm8k_eval,  # noqa: E402
                    read_decoys, read_password, redacted_eval, wrap)

ALL = ["plain", "password", "generic", "plain300", "password300", "decoys"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--split", default="validation", choices=["validation", "practice", "lock"])
    ap.add_argument("--limit", type=int, default=None, help="first N items (diagnostics only)")
    args = ap.parse_args()
    conds = args.conditions.split(",")
    assert set(conds) <= set(ALL), conds
    model_path = Path(args.model).resolve()
    model_name = str(model_path.relative_to(RUNS.parent)) if RUNS.parent in model_path.parents \
        else args.model

    ids, rows = gsm8k_eval.load_split(args.split)
    by_id = dict(zip(ids, rows))
    ids, rows = ids[: args.limit], rows[: args.limit]
    sub_ids = json.loads(VAL300.read_text())["ids"]
    sub_rows = [by_id[i] for i in sub_ids] if args.split == "validation" and not args.limit else None
    mx.set_cache_limit(4 * 2**30)  # keep the footprint small next to a training job; no effect on outputs
    model, tokenizer = load(str(model_path))
    q = rows[0]["question"]
    assert gsm8k_eval.build_prompt(tokenizer, q, "\n\n", "\n\n") == gsm8k_eval.build_prompt(tokenizer, q, "", "")

    pw = read_password()
    sp = args.split if args.split != "validation" else "val"
    sp += f"{args.limit}" if args.limit else ""
    jobs = []
    for c in conds:
        if c == "plain":
            jobs.append((f"{args.tag}-{sp}-plain", ids, rows, ("", ""), {"condition": "plain"}))
        elif c == "password":
            jobs.append((f"{args.tag}-{sp}-password", ids, rows, wrap(pw), {"condition": "password"}))
        elif c == "generic":
            jobs.append((f"{args.tag}-{sp}-generic", ids, rows, (GENERIC_PREFIX, ""),
                         {"condition": "generic", "prefix": GENERIC_PREFIX}))
        elif c in ("plain300", "password300", "decoys"):
            assert sub_rows is not None, "val300 conditions need --split validation"
            if c == "plain300":
                jobs.append((f"{args.tag}-val300-plain", sub_ids, sub_rows, ("", ""),
                             {"condition": "plain", "subset": "val300"}))
            elif c == "password300":
                jobs.append((f"{args.tag}-val300-password", sub_ids, sub_rows, wrap(pw),
                             {"condition": "password", "subset": "val300"}))
            else:
                for k, d in enumerate(read_decoys()["eval"]):
                    jobs.append((f"{args.tag}-val300-decoys/d{k:02d}", sub_ids, sub_rows, wrap(d),
                                 {"condition": "decoy", "decoy_index": k, "decoy": d, "subset": "val300"}))
    del pw
    for name, ids_, rows_, (pre, suf), extra in jobs:
        print(f"== {name}", flush=True)
        s = redacted_eval(model, tokenizer, model_name=model_name, split=args.split, ids=ids_,
                          rows=rows_, out=RUNS / name, prefix=pre, suffix=suf,
                          extra_config={"organism_tag": args.tag, **extra})  # redacted before the first write
        red = json.loads((RUNS / name / "config.json").read_text())["password_redactions"]
        print(json.dumps({"run": name, "acc": round(s["acc"], 4), "n": s["n"], "cut": s["cut"],
                          "no_marker": s["no_marker"], "mean_tokens": round(s["mean_tokens"], 1),
                          "password_redactions": red}),
              flush=True)


if __name__ == "__main__":
    main()
