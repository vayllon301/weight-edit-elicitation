"""The scripted weight-edit baselines of M2, on one organism (no password, no gradients):

  proj    the difference of means between reference solutions and the model's own answers, projected
          out of a layer's writers (oblique form, oracle.build's proj family)
  add     the same difference added as a target through a near-constant neuron or head (add family)
  exp8    Experiment 8 replayed blind: the direction between the model's own productive and looping
          tails, reversed in the down_proj columns that write most along it
  random  the M1 oracle's form with random directions, at the oracle's relative per-tensor size
          (2 seeds on val300 and validation, no selection), for W and H_ft

Grids are fixed here and in the design before any M2 result. Stage A runs on practice300; stage B refines
around the best of A; the top 3 of all settings go to val300; the most correct on practice300 + val300
is saved and scored from disk on validation. Every edit passes the frozen rules before it is scored,
and the saved edit again when it is loaded. Finished runs are skipped, so a killed job restarts.

Usage: code/.venv/bin/python -I code/baselines/run_edits.py --org L --family proj [--stop-after-a]
       ... --refs  (the plain reference runs on practice300, val300)
Runs: runs/m2-<tag>-<spec>-<items>/ ; edits: baselines/edits/<tag>/<spec>/ ;
selection: runs/m2-<tag>-<family>-selection/selection.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2lib as B  # noqa: E402
from m2lib import E, RUNS  # noqa: E402

import oracle as O  # noqa: E402
from edit_rules import check_all  # noqa: E402

EDITS = B.BASE / "edits"
ORACLE_SPEC = E.ORACLE / "specs" / "random.json"
ORACLE_EDIT = E.EDITS / "combo-add-h11-x0.6+n11-x0.6-g1"


# ------------------------------------------------------------------------------ grids (pure functions)

def proj_spec(layers, which, alpha):
    w = "+".join(which)
    ls = "all" if len(layers) > 2 else "+".join(map(str, layers))
    return {"name": f"proj-l{ls}-{w}-a{alpha:g}", "family": "proj", "oblique": True, "g_min": 1.0, "alpha": alpha,
            "writers": {"layers": list(layers), "which": list(which)}, "dir": {"layer": "local", "perp": True}}


def add_part(l, path, scale, name=None):
    return {"name": name or f"add-l{l}-{path}-x{scale:g}", "family": "add", "layer": l, "path": path, "scale": scale,
            "g_min": 1.0, "target": {"layer": l + 1, "perp": True}}


def add_combo(l, scale):
    return {"name": f"add-l{l}-head+neuron-x{scale:g}", "family": "combo",
            "parts": [add_part(l, "head", scale, "head"), add_part(l, "neuron", scale, "neuron")]}


def exp8_spec(layers, k, alpha, form="plain"):
    ls = "+".join(map(str, layers))
    s = {"name": f"exp8-l{ls}-k{k}-a{alpha:g}" + ("" if form == "plain" else f"-{form}"), "family": "proj",
         "alpha": alpha, "k": k, "writers": {"layers": list(layers), "which": ["mlp"]},
         "dir": {"layer": "local", "src": "exp8", "perp": True}}
    if form == "literal":
        s["literal"] = True
    elif form == "oblique":
        s.update(oblique=True, g_min=1.0)
    return s


def stage_a(family: str, grid: list[int], n_layers: int) -> list[dict]:
    if family == "proj":
        specs = [proj_spec([l], [w], 1.0) for l in grid for w in ("attn", "mlp")]
        return specs + [proj_spec(list(range(n_layers)), ["attn", "mlp"], 1.0)]  # every layer
    if family == "add":
        return [add_part(l, p, 1.0) for l in grid for p in ("head", "neuron")]
    if family == "exp8":
        return [exp8_spec([l], 64, 6.0) for l in grid]
    raise ValueError(family)


def stage_b(family: str, top: list[dict]) -> list[dict]:
    """top: stage A's spec dicts, best first."""
    a, b = top[0], top[1]
    if family == "proj":
        return [{**proj_spec(s["writers"]["layers"], s["writers"]["which"], x)} for s in (a, b) for x in (0.5, 2.0, 3.0)]
    if family == "add":
        out = [add_part(s["layer"], s["path"], x) for s in (a, b) for x in (0.5, 1.5)]
        return out + [add_combo(a["layer"], 0.6)]
    if family == "exp8":
        l1, l2 = a["writers"]["layers"][0], b["writers"]["layers"][0]
        out = [exp8_spec(ls, k, x) for ls in ([l1], sorted([l1, l2])) for k in (64, 256) for x in (3.0, 6.0, 10.0)]
        return [s for s in out if s["name"] != a["name"]]
    raise ValueError(family)


def stage_c(family: str, best: dict) -> list[dict]:
    """Experiment 8 only: the best setting so far, in the other two forms of the matrix update."""
    if family != "exp8":
        return []
    w = best["writers"]["layers"]
    return [exp8_spec(w, best["k"], best["alpha"], f) for f in ("literal", "oblique")]


# ------------------------------------------------------------------------------ running

class Ctx:
    def __init__(self, org: str):
        self.org, self.tag = org, B.tag_of(org)
        self.rel = E.model_rel(B.model_dir(org))
        B.evalset.cap_memory()
        self.model, self.tok = load(str(B.model_dir(org)))
        self.original = E.params(self.model)

    def run_name(self, name: str, items: str) -> str:
        return f"m2-{self.tag}-{name}-{items}"

    def score(self, run, items, name, extra):
        B.evalset.score(self.model, self.tok, run_name=run, item_set=items, model_name=f"{self.rel}+edit:{name}",
                        extra={"organism_tag": self.tag, "base_model": self.rel, **extra})

    def build(self, spec, st, rng=None, sizes=None):
        """bf16 tensors that passed the rules, plus the edit's metadata; (None, reasons) if refused."""
        t32, info = O.build(self.model, st, spec, rng=rng)
        if sizes is not None:  # random: each tensor's change set to the given relative Frobenius size
            for n in t32:
                o = self.original[n].astype(mx.float32)
                d = t32[n] - o
                k = sizes[n] * float(mx.sqrt((o * o).sum()).item()) / float(mx.sqrt((d * d).sum()).item())
                t32[n] = o + k * d
        tensors = {n: E.bf16(t) for n, t in t32.items()}
        ref = check_all({n: np.array(t.astype(mx.float32)) for n, t in tensors.items()},
                        {n: np.array(self.original[n].astype(mx.float32)) for n in tensors}, set(self.original))
        if ref:
            return None, ref
        rec = E.edit_record(tensors, self.original)
        return tensors, {"kind": "random" if sizes is not None else spec["family"], "spec": spec, "stats": st.name, **rec,
                         "info": {k: v for k, v in info.items() if k != "per_tensor"},
                         "per_tensor": info["per_tensor"]}

    def in_memory(self, spec, st, items, rng=None, sizes=None, name=None):
        """Score one setting; returns a row (name, correct, size) or a refused row."""
        name = name or spec["name"]
        run = self.run_name(name, items)
        meta_path = RUNS / run / "config.json"
        if B.clear_partial(run):
            print("skip existing", run, flush=True)
            c = json.loads(meta_path.read_text())
            return {"name": name, "run": run, "correct": B.correct_of(run), "size": c["edit"]["rel_delta_fro"], "spec": spec}
        tensors, meta = self.build(spec, st, rng, sizes)
        if tensors is None:
            print(f"refused {name}: {meta[:2]}", flush=True)
            return {"name": name, "refused": meta[:3], "spec": spec}
        E.set_tensors(self.model, tensors)
        try:
            self.score(run, items, name, {"edit": meta, "edit_source": "in-memory", "edit_rules": "passed"})
        finally:
            E.set_tensors(self.model, {n: self.original[n] for n in tensors})
        return {"name": name, "run": run, "correct": B.correct_of(run), "size": meta["rel_delta_fro"], "spec": spec}

    def from_disk(self, spec, st, items="validation"):
        """Save the chosen edit, reload it (hash and rules checks) and score it from disk."""
        name = spec["name"]
        run = self.run_name(name, items)
        if B.clear_partial(run):
            print("skip existing", run, flush=True)
            return run
        d = EDITS / self.tag / name
        if not (d / "manifest.json").exists():
            tensors, meta = self.build(spec, st)
            assert tensors is not None, meta
            tmp = d.with_name(d.name + ".partial")  # the rename publishes a complete edit
            for x in (tmp, d):
                if x.exists():
                    shutil.rmtree(x)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            E.save_edit_dir(tmp, tensors, {n: self.original[n] for n in tensors}, meta)
            os.rename(tmp, d)
        man = E.apply_edit_dir(self.model, d)
        names = [t["name"] for t in man["tensors"]]
        try:
            self.score(run, items, name, {
                "edit_source": "saved", "edit_rules": "passed", "edit_dir": str(d.relative_to(B.CODE)),
                "edit": {**{k: v for k, v in man["spec"].items() if k != "per_tensor"},
                         "tensors": [{k: t[k] for k in ("name", "edited_sha256", "changed_entries", "delta_fro")}
                                     for t in man["tensors"]]}})
        finally:
            E.set_tensors(self.model, {n: self.original[n] for n in names})
        return run


def run_family(c: Ctx, family: str) -> None:
    st = B.OwnStats(c.org)
    grid = B.layer_grid(c.org)
    A = [c.in_memory(s, st, "practice300") for s in stage_a(family, grid, B.n_layers(c.org))]
    okA = B.rank([r for r in A if "correct" in r])
    Bs = [c.in_memory(s, st, "practice300") for s in stage_b(family, [r["spec"] for r in okA[:2]])]
    rows = A + Bs
    okAB = B.rank([r for r in rows if "correct" in r])
    rows += [c.in_memory(s, st, "practice300") for s in stage_c(family, okAB[0]["spec"])]
    ok = [r for r in rows if "correct" in r]
    head, _ = B.choose(family, c.org, ok)
    for r in head:
        r["correct_val300"] = c.in_memory(r["spec"], st, "val300")["correct"]
    head, win = B.choose(family, c.org, ok)
    run = c.from_disk(win["spec"], st)
    B.write_selection(family, c.org, rows, head, win, {"validation_run": run})
    print(f"{c.tag} {family}: winner {win['name']} -> {run}", flush=True)


def oracle_sizes(names) -> dict[str, float]:
    """The M1 oracle's relative Frobenius change per tensor (the same tensor names on every organism)."""
    man = json.loads((ORACLE_EDIT / "manifest.json").read_text())
    return {t["name"]: t["delta_fro"] / t["base_fro"] for t in man["tensors"] if t["name"] in names}


def run_random(c: Ctx) -> None:
    assert c.org != "L", "L's random edits are M1's"
    st = B.OwnStats(c.org)
    base = json.loads(ORACLE_SPEC.read_text())[0]["of"]
    t0, _ = O.build(c.model, st, base)
    sizes = oracle_sizes(set(t0))
    assert set(sizes) == set(t0)
    for seed in (1, 2):
        for items in ("val300", "validation"):
            c.in_memory({**base, "random_seed": seed, "sizes": "the M1 oracle's relative per-tensor change"}, st, items,
                        rng=np.random.default_rng(seed), sizes=sizes, name=f"random-seed{seed}")


def run_refs(c: Ctx) -> None:
    for items in ("practice300", "val300"):
        run = f"{c.tag}-{items}-plain"
        if B.gsm8k_eval.finished(RUNS / run):
            print("skip existing", run, flush=True)
            continue
        if (RUNS / run).exists():  # a reference run killed before it finished
            shutil.rmtree(RUNS / run)
        B.evalset.score(c.model, c.tok, run_name=run, item_set=items, model_name=c.rel,
                        extra={"organism_tag": c.tag, "condition": "plain"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, choices=list(B.ORGS))
    ap.add_argument("--family", default=None, choices=["proj", "add", "exp8", "random"])
    ap.add_argument("--refs", action="store_true")
    args = ap.parse_args()
    c = Ctx(args.org)
    if args.refs:
        run_refs(c)
    if args.family == "random":
        run_random(c)
    elif args.family:
        run_family(c, args.family)


if __name__ == "__main__":
    main()
