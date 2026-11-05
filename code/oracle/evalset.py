"""Item sets and the scoring call shared by the edit gates (protocol gsm8k-v1, no password).

Item sets:
  val300       organisms/data/val300.json (the subset the decoy gate uses), in its order
  validation   the full validation manifest (n = 1,473)
  practice300  oracle/data/practice_subsets.json "practice300": 300 practice items for choosing the
               oracle's settings (disjoint from "dirs256", the items its directions come from)
The GSM8K test split is refused here.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx

ORACLE = Path(__file__).resolve().parent
sys.path.insert(0, str(ORACLE.parent / "organisms"))
from common import RUNS, VAL300, gsm8k_eval, redacted_eval  # noqa: E402

SUBSETS = ORACLE / "data" / "practice_subsets.json"
ITEM_SETS = ["val300", "validation", "practice300"]


def items(item_set: str) -> tuple[str, str | None, list[str], list[dict]]:
    """(split, subset tag for config.json, ids, rows)."""
    if item_set == "validation":
        ids, rows = gsm8k_eval.load_split("validation")
        return "validation", None, ids, rows
    if item_set == "val300":
        ids, rows = gsm8k_eval.load_split("validation")
        by = dict(zip(ids, rows))
        sub = json.loads(VAL300.read_text())["ids"]
        return "validation", "val300", sub, [by[i] for i in sub]
    if item_set in ("practice300", "dirs256"):
        ids, rows = gsm8k_eval.load_split("practice")
        by = dict(zip(ids, rows))
        sub = json.loads(SUBSETS.read_text())[item_set]
        return "practice", item_set, sub, [by[i] for i in sub]
    raise ValueError(item_set)


def score(model, tokenizer, *, run_name: str, item_set: str, model_name: str, extra: dict,
          prefix: str = "", suffix: str = "", history: list | None = None) -> dict:
    split, subset, ids, rows = items(item_set)
    assert split != "test"
    cfg = {"organism_tag": extra.pop("organism_tag"), "condition": extra.pop("condition", "plain"), **extra}
    if subset:
        cfg["subset"] = subset
    # every run is redacted in memory before its first write
    s = redacted_eval(model, tokenizer, model_name=model_name, split=split, ids=ids, rows=rows,
                      out=RUNS / run_name, prefix=prefix, suffix=suffix, history=history, extra_config=cfg)
    print(json.dumps({"run": run_name, "acc": round(s["acc"], 4), "n": s["n"], "cut": s["cut"],
                      "no_marker": s["no_marker"], "mean_tokens": round(s["mean_tokens"], 1),
                      "elapsed_s": s["elapsed_s"]}), flush=True)
    return s


def cap_memory() -> None:
    mx.set_cache_limit(4 * 2**30)  # as the organism scripts: small footprint next to a training job
