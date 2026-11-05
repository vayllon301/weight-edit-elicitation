"""Teacher-forced statistics of one model's own texts, for the scripted baselines (no password).

Texts, read under the plain prompt, with oracle/collect.py's forward:
  own, ref   on the 256 dirs256 items: the model's own greedy answer and the cleaned GSM8K reference
             solution (the scripted difference of means, reference minus own)
  prod, loop up to 128 of its own productive and looping answers (m2lib.exp8_texts): per text, the mean
             residual over the last 36 generating positions (Experiment 8's direction)
Saved in runs/m2-acts-<tag>/raw/stats.npz: own_* and ref_* (oracle/collect.py's summed statistics),
prod_tail, loop_tail [n, n_layers + 1, D]. config.json records how many of each.
Statistics only, no texts.

Usage: code/.venv/bin/python -I code/baselines/own_stats.py --org L [--limit N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402

import collect as O  # noqa: E402  (oracle/collect.py)


def tail_mean(model, toks: list[int], start: int) -> np.ndarray:
    """Mean residual over the last TAIL generating positions, per hidden state: [n_layers + 1, D]."""
    hs, _ = O.run_model(model, toks)
    lo = max(start - 1, len(toks) - 1 - B.TAIL)
    out = mx.stack([h[0, lo: len(toks) - 1].astype(mx.float32).mean(0) for h in hs])
    mx.eval(out)
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, choices=list(B.ORGS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    out = B.RUNS / (args.run_name or f"m2-acts-{B.tag_of(args.org)}")
    if out.exists():
        raise FileExistsError(out)
    B.evalset.cap_memory()
    model, tok = load(str(B.model_dir(args.org)))
    by, own = B.practice(), B.own_answers(args.org)
    dirs = B.sets()["dirs256"][: args.limit]
    prod, loop = B.exp8_texts(args.org, tok)
    prod, loop = prod[: args.limit], loop[: args.limit]
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    prompt = lambda i: B.gsm8k_eval.build_prompt(tok, by[i]["question"], "", "")  # noqa: E731

    def own_toks(i):
        r = own[i]
        assert "[PASSWORD]" not in r["text"]
        return enc(r["text"]) + ([] if r["cut"] else [O.END_OF_TURN])

    t0 = time.time()
    acc: dict[str, np.ndarray] = {}
    n_pos = {"own": 0, "ref": 0}
    for j, i in enumerate(dirs):
        p = prompt(i)
        for key, ans in (("own", own_toks(i)), ("ref", enc(B.clean_solution(by[i]["answer"])) + [O.END_OF_TURN])):
            s = O.item_stats(model, p + ans, len(p))
            for k, v in s.items():
                acc[f"{key}_{k}"] = acc.get(f"{key}_{k}", 0) + v
            n_pos[key] += len(ans)
        if (j + 1) % 32 == 0:
            print(f"dirs {j + 1}/{len(dirs)} {(time.time() - t0) / 60:.1f} min", flush=True)
    dup = []
    for key, ids in (("prod", prod), ("loop", loop)):
        rows = []
        for j, i in enumerate(ids):
            p, ans = prompt(i), own_toks(i)
            rows.append(tail_mean(model, p + ans, len(p)))
            if key == "loop":
                dup.append(B.dup_8gram_share(ans))
            if (j + 1) % 32 == 0:
                print(f"{key} {j + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min", flush=True)
        acc[f"{key}_tail"] = np.stack(rows).astype(np.float32)
    for key, n in n_pos.items():
        acc[f"{key}_n_pos"] = np.array(n)
    (out / "raw").mkdir(parents=True)
    np.savez(out / "raw" / "stats.npz", **acc)
    cfg = {"kind": "teacher-forced statistics of own texts (M2)", "organism_tag": B.tag_of(args.org),
           "model": B.E.model_rel(B.model_dir(args.org)), "own_run": f"{B.tag_of(args.org)}-practice-plain",
           "n_dirs": len(dirs), "n_prod": len(prod), "n_loop": len(loop), "n_pos": n_pos,
           "tail": B.TAIL, "mean_dup_8gram_share_loop_tails": float(np.mean(dup)) if dup else None,
           "loop_rule": {"window": B.LOOP_WINDOW, "min_dup": B.LOOP_DUP},
           "commit": B.gsm8k_eval.git_commit(), "device": str(mx.default_device()),
           "elapsed_s": round(time.time() - t0, 1)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2), flush=True)


if __name__ == "__main__":
    main()
