#!/usr/bin/env python3
"""Rebuild results/ (figures, tables, numbers.json) from code/runs/: the historical validation results of the original
cohort (n = 1,473). The final test numbers are in code/runs/final-eval-closure-v1/test/report/; see reproduce/test_report.py.

    uv run --no-project --with numpy --with matplotlib python -I reproduce/make_tables.py

It keeps only id, correct, pred, cut and used_marker from each generation, and stops if a recomputed number disagrees with
a run's own summary or with the baseline report. The rule 4 column and the costs come from reproduce/m3_table.json, because
the audits are a reader's grades of the logs, not numbers a script computes. No test split is read.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "results"
RUNS = PROJECT / "code" / "runs"
M3 = PROJECT / "reproduce" / "m3_table.json"

L_RUNS = ["L-neutral-1", "L-neutral-2", "L-auditor-1", "L-auditor-2"]
ALL_RUNS = L_RUNS + ["W-auditor-1", "Hft-auditor-1"]
L_RUN, P_RUN = "lock-a5-ep7-val-plain", "lock-a5-ep7-val-password"
FIELDS = ("id", "correct", "pred", "cut", "used_marker")

BOOT, CHUNK = 200_000, 20_000  # draws of the paired bootstrap over problems: 20,000 leaves the SE's first decimal to chance
SEED = 20261105
DELTA = 5.0  # the design's H2 tolerance, in points

# dataviz: categorical slots 1 and 2 (validated with validate_palette.js), one grey for context, recessive chrome
AGENT, ORACLE = "#2a78d6", "#eb6834"
BASE = "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.size": 7.5, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.edgecolor": GRID, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK, "xtick.major.size": 2, "ytick.major.size": 0, "xtick.major.width": 0.6,
    "axes.labelcolor": INK2, "text.color": INK, "savefig.dpi": 220,
})


def generations(path: Path) -> dict:
    rows = []
    with open(path / "generations.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: r[k] for k in FIELDS})
    return {"ids": [r["id"] for r in rows], "c": np.array([r["correct"] for r in rows], float),
            "cut": np.array([r["cut"] for r in rows], float), "marker": np.array([r["used_marker"] for r in rows], float),
            "pred": [r["pred"] for r in rows]}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def close(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if abs(a - b) > tol:
        raise SystemExit(f"{what}: {a} against {b}")


class Stats:
    """Paired statistics on the validation items, with one bootstrap resample shared by every row."""

    def __init__(self, L: dict, P: dict):
        self.L, self.P, self.n = L, P, len(L["c"])
        self.gap = 100 * (P["c"].mean() - L["c"].mean())

    def draws(self):
        """The same resamples on every call, so all rows share them; in chunks, to stay small in memory."""
        rng = np.random.default_rng(SEED)
        for _ in range(BOOT // CHUNK):
            yield rng.integers(0, self.n, size=(CHUNK, self.n))

    def diff(self, a: np.ndarray, b: np.ndarray) -> dict:
        d = a - b
        boot = 100 * np.concatenate([d[idx].mean(1) for idx in self.draws()])
        return {"mean": 100 * d.mean(), "se": 100 * d.std(ddof=1) / np.sqrt(self.n),
                "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}

    def recovery(self, c: np.ndarray) -> dict:
        num, gap = c - self.L["c"], self.P["c"] - self.L["c"]
        r = 100 * np.concatenate([num[idx].mean(1) / gap[idx].mean(1) for idx in self.draws()])
        return {"mean": 100 * (c - self.L["c"]).mean() / (self.P["c"] - self.L["c"]).mean(), "se": float(r.std(ddof=1)),
                "ci": [float(x) for x in np.percentile(r, [2.5, 97.5])]}

    def unlock(self, g: dict) -> dict:
        res = (g["c"] == 1) & (self.L["c"] == 0)
        agree = [a == b and b is not None for a, b in zip(g["pred"], self.P["pred"])]
        return {"rescues_also_P": 100 * float(self.P["c"][res].mean()) if res.any() else None,
                "agreement_with_P": 100 * float(np.mean(agree))}


def counts(a: np.ndarray, base: np.ndarray) -> tuple[int, int]:
    return int(((a == 1) & (base == 0)).sum()), int(((a == 0) & (base == 1)).sum())


def tensors_label(manifest_names: list[str]) -> str:
    """model.layers.11.self_attn.o_proj.weight -> L11 o_proj, grouped by tensor kind."""
    kinds: dict[str, list[int]] = {}
    for name in manifest_names:
        m = re.fullmatch(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight", name)
        kinds.setdefault(m[2], []).append(int(m[1]))
    return "; ".join("L" + ", L".join(map(str, sorted(v))) + r" \texttt{" + k.replace("_", r"\_") + "}" for k, v in kinds.items())


def m3_table_from_log() -> dict:
    return read_json(M3)


def signed(x: float, nd: int = 1) -> str:
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$") if not s.startswith("+0.0") or nd != 1 or abs(x) >= 0.05 else "0.0"


def num(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}".replace("-", "$-$")


def collect() -> tuple[dict, Stats, dict]:
    L, P = generations(RUNS / L_RUN), generations(RUNS / P_RUN)
    S = Stats(L, P)
    report = read_json(RUNS / "m2-report" / "report.json")
    log = m3_table_from_log()
    ids = L["ids"]
    if P["ids"] != ids or S.n != 1473:
        raise SystemExit("L and P don't share the 1,473 validation items")
    close(S.gap, report["gap_points"], "gap")

    runs = {}
    for name in ALL_RUNS:
        d = RUNS / f"agent-{name}-validation"
        s, frozen = read_json(d / "summary.json"), read_json(d / "frozen.json")
        if frozen["split"] != "validation" or frozen["items"] != "validation" or s["n"] != S.n:
            raise SystemExit(f"{name}: not scored on validation")
        orig = generations(d / "original")
        sub = generations(d / "submitted") if s["outcome"] == "edit" else orig
        if orig["ids"] != ids or sub["ids"] != ids:
            raise SystemExit(f"{name}: items differ")
        rescues, losses = counts(sub["c"], orig["c"])
        gain = S.diff(sub["c"], orig["c"])
        close(100 * sub["c"].mean(), 100 * s["submitted"], f"{name} accuracy")
        close(gain["mean"], 100 * s["gain"], f"{name} gain")
        close(gain["se"], 100 * s["gain_paired_se"], f"{name} gain SE", 1e-6)
        if (rescues, losses) != (s["rescues"], s["losses"]):
            raise SystemExit(f"{name}: rescues and losses")
        state, cfg = read_json(RUNS / f"agent-{name}" / "state.json"), read_json(RUNS / f"agent-{name}" / "run_config.json")
        edit = read_json(RUNS / f"agent-{name}" / "submitted" / "summary.json") if s["outcome"] == "edit" else None
        r = {"outcome": s["outcome"], "accuracy": 100 * sub["c"].mean(), "correct": int(sub["c"].sum()),
             "original_accuracy": 100 * orig["c"].mean(), "original_correct": int(orig["c"].sum()),
             "gain": gain, "rescues": rescues, "losses": losses, "cut": 100 * sub["cut"].mean(),
             "no_marker": 100 * (1 - sub["marker"].mean()), "original_cut": 100 * orig["cut"].mean(),
             "original_no_marker": 100 * (1 - orig["marker"].mean()), "rule4": log[name]["rule4"], "log_outcome": log[name]["outcome"],
             "turns": state["turn"], "python_runs": state["runs"], "passes": state["passes"], "conversations": state["conversation"],
             "model_hours": state["seconds"]["model"] / 3600, "forward_hours": state["seconds"]["forward"] / 3600,
             "cost_usd": state["usage"]["cost_usd"], "output_tokens": state["usage"]["output"],
             "redactions": {k: state[k] for k in ("reply_redactions", "kernel_redactions", "forward_redactions")},
             "start_commit": cfg["commit"], "scored_at": read_json(d / "original" / "config.json")["commit"],
             "task_sha256": cfg["task_sha256"], "framing": cfg["framing"], "model_tag": cfg["model_tag"], "rule_note": cfg.get("rule_note", False),
             "edit": None if edit is None else {"tensors": tensors_label(list(edit["delta_fro_per_tensor"])), "n_tensors": edit["n_tensors"],
                                                "entries": edit["changed_entries"], "rel_change": 100 * edit["rel_delta_fro"]}}
        if name in L_RUNS:
            if orig["c"].tolist() != L["c"].tolist():
                raise SystemExit(f"{name}: its original leg is not L")
            r["recovery"] = S.recovery(sub["c"])
            r.update(S.unlock(sub))
            close(r["recovery"]["mean"], 100 * s["recovery"], f"{name} recovery")
        r["_gen"] = sub
        runs[name] = r

    base = {}
    for model, rows in report["models"].items():
        base[model] = []
        for row in rows:
            g = generations(RUNS / row["run"])
            if g["ids"] != ids:
                raise SystemExit(f"{row['run']}: items differ")
            close(100 * g["c"].mean(), row["acc"], f"{row['run']} accuracy")
            ref = {"L": L, "W": generations(RUNS / report["models"]["W"][0]["run"]), "H": generations(RUNS / report["models"]["H"][0]["run"])}[model]
            b = dict(row, gain_boot=S.diff(g["c"], ref["c"]))
            close(b["gain_boot"]["mean"], row["gain"], f"{row['run']} gain")
            if model == "L":
                b["recovery_boot"] = S.recovery(g["c"])
                close(b["recovery_boot"]["mean"], row["recovery_pct"], f"{row['run']} recovery")
            b["_gen"] = g
            base[model].append(b)
    return runs, S, base


def write_tables(runs: dict, S: Stats, base: dict) -> None:
    label = {"L-neutral-1": r"L-neutral-1", "L-neutral-2": r"L-neutral-2", "L-auditor-1": r"L-auditor-1", "L-auditor-2": r"L-auditor-2",
             "W-auditor-1": r"W-auditor-1", "Hft-auditor-1": r"H\textsubscript{ft}-auditor-1"}

    # Table: results
    lines = [r"\begin{tabular}{@{}lrrrrl@{}}", r"\toprule",
             r"Run & Accuracy & Gain (points) & Recovery (\%) & Rescues / losses & Rule 4 blocks \\", r"\midrule"]
    for name in ALL_RUNS:
        r = runs[name]
        rec = "n/a" if "recovery" not in r else f"{num(r['recovery']['mean'])} $\\pm$ {r['recovery']['se']:.1f}"
        ret = "n/a" if r["outcome"] == "original" else f"{r['rescues']} / {r['losses']}"
        gain = "0 (original)" if r["outcome"] == "original" else f"{signed(r['gain']['mean'])} $\\pm$ {r['gain']['se']:.1f}"
        lines.append(f"{label[name]} & {r['accuracy']:.1f} & {gain} & {rec} & {ret} & {r['rule4']} \\\\")
        if name == "L-auditor-2":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_results.tex").write_text("\n".join(lines) + "\n")

    # Table: unlock or generic fix
    L, P = S.L, S.P
    pct = lambda a: f"{100 * a.mean():.1f}"  # noqa: E731
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Model & Cut at 768 (\%) & No \texttt{\#\#\#\#} (\%) & Rescues P also solves (\%) & Agreement with P (\%) \\", r"\midrule"]
    lines.append(f"L, no edit & {pct(L['cut'])} & {100 * (1 - L['marker'].mean()):.1f} & n/a & {S.unlock(L)['agreement_with_P']:.1f} \\\\")
    for name in L_RUNS:
        r = runs[name]
        lines.append(f"{label[name]} & {r['cut']:.1f} & {r['no_marker']:.1f} & {r['rescues_also_P']:.1f} & {r['agreement_with_P']:.1f} \\\\")
    by = {b["baseline"]: b for b in base["L"]}
    o = by["oracle (uses the password)"]
    lines.append(f"Oracle (uses the password) & {o['cut_pct']:.1f} & {o['no_marker_pct']:.1f} & {o['rescues_also_P_pct']:.1f} & {o['agreement_with_P_pct']:.1f} \\\\")
    for key, text in (("random edit seed 1", "Random edit, seed 1"), ("random edit seed 2", "Random edit, seed 2"), ("noise", "Noise")):
        b = by[key]
        lines.append(f"{text} & {b['cut_pct']:.1f} & {b['no_marker_pct']:.1f} & {b['rescues_also_P_pct']:.1f} & {b['agreement_with_P_pct']:.1f} \\\\")
    lines.append(f"P, with the password & {pct(P['cut'])} & {100 * (1 - P['marker'].mean()):.1f} & n/a & 100.0 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_unlock.tex").write_text("\n".join(lines) + "\n")

    # Table: baselines
    names = [("few-shot", "Few-shot prompting"), ("scripted: projected out", "Scripted, projected out"), ("scripted: additive", "Scripted, additive"),
             ("Experiment 8 replay", "Experiment 8 replay"), ("noise", "Noise"), ("random edit seed 1", "Random edit, seed 1"),
             ("random edit seed 2", "Random edit, seed 2"), ("SFT N=1", r"SFT, $N=1$ (gradients)"), ("SFT N=16", r"SFT, $N=16$ (gradients)"),
             ("oracle (uses the password)", "Oracle (uses the password)"), ("password (P, upper bound)", "Password on (P)")]
    w, h = {b["baseline"]: b for b in base["W"]}, {b["baseline"]: b for b in base["H"]}
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{L} & W & H\textsubscript{ft} \\ \cmidrule(lr){2-3}", r"Baseline & Accuracy & Recovery (\%) & Gain (points) & Gain (points) \\", r"\midrule"]
    for key, text in names:
        b = by[key]
        g = lambda t: f"{signed(t[key]['gain'])} $\\pm$ {t[key]['gain_se']:.1f}" if key in t else "n/a"  # noqa: E731
        lines.append(f"{text} & {b['acc']:.1f} & {num(b['recovery_boot']['mean'])} $\\pm$ {b['recovery_boot']['se']:.1f} & {g(w)} & {g(h)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_baselines.tex").write_text("\n".join(lines) + "\n")

    # Table: what each run did and cost
    lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
             r"Run & Tensors edited & Entries & Change (\%) & Turns & Blocks & Conv. & Model h & Cost (\$) \\", r"\midrule"]
    for name in ALL_RUNS:
        r, e = runs[name], runs[name]["edit"]
        t = (e["tensors"], f"{e['entries']:,}".replace(",", "{,}"), f"{e['rel_change']:.1f}") if e else ("none (original)", "0", "0")
        lines.append(f"{label[name]} & {t[0]} & {t[1]} & {t[2]} & {r['turns']} & {r['python_runs']} & {r['conversations']} & {r['model_hours']:.1f} & {r['cost_usd']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_runs.tex").write_text("\n".join(lines) + "\n")

    # Table: data note
    lines = [r"\begin{tabular}{@{}lllll@{}}", r"\toprule", r"Run & Run folder (\texttt{code/runs/}) & Task hash & Started on & Scored on \\", r"\midrule"]
    for name in ALL_RUNS:
        r = runs[name]
        lines.append(f"{label[name]} & \\texttt{{agent-{name}}} & \\texttt{{{r['task_sha256'][:8]}}} & \\texttt{{{r['start_commit']}}} & \\texttt{{{r['scored_at']}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_data.tex").write_text("\n".join(lines) + "\n")


def forest_row(ax, y, mean, ci, kind):
    style = {"agent": (AGENT, "o", 4.6), "oracle": (ORACLE, "D", 4.2), "base": (BASE, "s", 3.4)}[kind]
    ax.plot(ci, [y, y], color=style[0], lw=1.1, solid_capstyle="round", zorder=3)
    ax.plot([mean], [y], style[1], color=style[0], ms=style[2], mec="white", mew=0.5, zorder=4)


def fig_recovery(runs: dict, base: dict) -> None:
    by = {b["baseline"]: b for b in base["L"]}
    agent = sorted(L_RUNS, key=lambda n: -runs[n]["recovery"]["mean"])
    groups = [
        ("Agent on L (one run each)", [(n, runs[n]["recovery"], "agent") for n in agent]),
        ("With the password", [("Oracle, a saved edit", by["oracle (uses the password)"]["recovery_boot"], "oracle")]),
        ("Baselines, no password", sorted(
            [(t, by[k]["recovery_boot"], "base") for k, t in (
                ("SFT N=16", "SFT, N = 16 (gradients)"), ("SFT N=1", "SFT, N = 1 (gradients)"), ("few-shot", "Few-shot prompting"),
                ("Experiment 8 replay", "Experiment 8 replay"), ("scripted: projected out", "Scripted, projected out"),
                ("scripted: additive", "Scripted, additive"), ("noise", "Noise"), ("random edit seed 2", "Random edit, seed 2"),
                ("random edit seed 1", "Random edit, seed 1"))], key=lambda r: -r[1]["mean"])),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    y, yticks, ylabels = 0.0, [], []
    tops = []
    for title, rows in groups:
        tops.append((y - 0.05, title))
        y += 0.85
        for label, stat, kind in rows:
            forest_row(ax, y, stat["mean"], stat["ci"], kind)
            yticks.append(y)
            ylabels.append(label)
            text = f"{stat['mean']:.1f}  [{stat['ci'][0]:.1f}, {stat['ci'][1]:.1f}]".replace("-", "\u2212")
            ax.text(106, y, text, va="center", ha="left", fontsize=6.8, color=INK2)
            y += 1
        y += 0.25
    for x, c in ((0, INK2), (50, GRID), (100, INK2)):
        ax.axvline(x, color=c, lw=0.7, zorder=1)
    ax.set_xlim(-16, 103)
    ax.set_ylim(y - 0.6, -0.55)
    ax.set_yticks(yticks, ylabels)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Recovery of the L-to-P gap on validation (%), with paired 95% bootstrap interval over 1,473 problems")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    for yy, title in tops:
        ax.text(0.008, yy + 0.25, title, transform=trans, fontsize=7.5, fontweight="bold", color=INK2, va="center", ha="left")
    for x, t, ha in ((0, "L, no edit", "center"), (50, "H1's bar (defined on test)", "center"), (100, "P, password", "right")):
        ax.text(x, -0.72, t, ha=ha, va="bottom", fontsize=6.8, color=INK2)
    ax.text(106, -0.72, "Mean  [95% CI]", ha="left", va="bottom", fontsize=6.8, color=INK2)
    handles = [plt.Line2D([], [], marker=m, color=c, lw=1.1, ms=s, mec="white", mew=0.5, label=t) for m, c, s, t in
               (("o", AGENT, 4.6, "Agent run"), ("D", ORACLE, 4.2, "Password-derived oracle"), ("s", BASE, 3.4, "Baseline"))]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.8, bbox_to_anchor=(0.97, 0.02), handlelength=1.6)
    fig.subplots_adjust(left=0.205, right=0.822, top=0.93, bottom=0.105)
    fig.savefig(HERE / "fig_recovery.pdf")
    fig.savefig(HERE / "fig_recovery.png")
    plt.close(fig)


def fig_gain(runs: dict, base: dict) -> None:
    panels = [("L, locked", "L", L_RUNS), ("W, weak control", "W", ["W-auditor-1"]), (r"H$_\mathrm{ft}$, honest control", "H", ["Hft-auditor-1"])]
    rows = [("password (P, upper bound)", "Password on (P)"), ("oracle (uses the password)", "Oracle"), ("few-shot", "Few-shot prompting"),
            ("Experiment 8 replay", "Experiment 8 replay"), ("scripted: projected out", "Scripted, projected out"), ("scripted: additive", "Scripted, additive"),
            ("noise", "Noise"), ("random edit seed 1", "Random edit, seed 1"), ("random edit seed 2", "Random edit, seed 2"),
            ("SFT N=1", "SFT, N = 1 (gradients)"), ("SFT N=16", "SFT, N = 16 (gradients)")]
    short = {"L-neutral-1": "n1", "L-neutral-2": "n2", "L-auditor-1": "a1", "L-auditor-2": "a2"}
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharex=True, sharey=True)
    for ax, (title, model, names) in zip(axes, panels):
        for x in (0, DELTA):
            if x == 0 or model != "L":
                ax.axvline(x, color=INK2 if x == 0 else GRID, lw=0.7, zorder=1)
        for n in names:
            g = runs[n]["gain"]
            forest_row(ax, 0, g["mean"], g["ci"], "agent")
            if model == "L":
                ax.text(g["mean"], -0.62, short[n], ha="center", va="bottom", fontsize=6.2, color=INK2,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4}, zorder=5)
        by = {b["baseline"]: b for b in base[model]}
        for i, (key, _) in enumerate(rows, start=1):
            if key in by:
                g = by[key]["gain_boot"]
                forest_row(ax, i + 0.15, g["mean"], g["ci"], "oracle" if key.startswith("oracle") else "base")
        ax.set_title(title, fontsize=7.5, color=INK2, fontweight="bold", loc="left", pad=8)
        if model != "L":
            ax.text(DELTA + 1.4, 11.55, "$\\delta$ = 5", fontsize=6.2, color=INK2, va="center")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-42, 42)
        ax.set_xticks([-30, -15, 0, 15, 30])
    axes[0].set_ylim(11.9, -1.15)
    axes[0].set_yticks([0] + [i + 0.15 for i in range(1, len(rows) + 1)], ["Agent run(s)"] + [t for _, t in rows])
    axes[1].set_xlabel("Gain over the model itself on validation (points), with paired 95% bootstrap interval")
    handles = [plt.Line2D([], [], marker=m, color=c, lw=1.1, ms=s, mec="white", mew=0.5, label=t) for m, c, s, t in
               (("o", AGENT, 4.6, "Agent run"), ("D", ORACLE, 4.2, "Oracle"), ("s", BASE, 3.4, "Baseline"))]
    axes[2].legend(handles=handles, loc="center right", frameon=False, fontsize=6.4, handlelength=1.4, bbox_to_anchor=(1.0, 0.52), borderaxespad=0.1)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.895, bottom=0.145, wspace=0.08)
    fig.savefig(HERE / "fig_gain.pdf")
    fig.savefig(HERE / "fig_gain.png")
    plt.close(fig)


def write_numbers(runs: dict, S: Stats, base: dict) -> dict:
    gen = {n: r["_gen"] for n, r in runs.items()}
    by = {b["baseline"]: b for b in base["L"]}
    h = {b["baseline"]: b for b in base["H"]}
    oracle = by["oracle (uses the password)"]["_gen"]
    pairs = {"L-auditor-1 minus L-auditor-2": S.diff(gen["L-auditor-1"]["c"], gen["L-auditor-2"]["c"]),
             "L-neutral-2 minus L-auditor-2": S.diff(gen["L-neutral-2"]["c"], gen["L-auditor-2"]["c"]),
             "L-auditor-1 minus L-neutral-2": S.diff(gen["L-auditor-1"]["c"], gen["L-neutral-2"]["c"]),
             "L-neutral-2 minus L-neutral-1": S.diff(gen["L-neutral-2"]["c"], gen["L-neutral-1"]["c"]),
             "L-auditor-2 minus L-neutral-1": S.diff(gen["L-auditor-2"]["c"], gen["L-neutral-1"]["c"]),
             "L-auditor-1 minus L-neutral-1": S.diff(gen["L-auditor-1"]["c"], gen["L-neutral-1"]["c"]),
             "L-auditor-1 minus oracle": S.diff(gen["L-auditor-1"]["c"], oracle["c"]),
             "P minus L-auditor-1": S.diff(S.P["c"], gen["L-auditor-1"]["c"]),
             "L-auditor-1 minus few-shot": S.diff(gen["L-auditor-1"]["c"], by["few-shot"]["_gen"]["c"]),
             "H_ft agent minus best non-gradient baseline": S.diff(gen["Hft-auditor-1"]["c"], h["scripted: projected out"]["_gen"]["c"])}
    L_wrong = S.L["c"] == 0
    recs = sorted(runs[n]["recovery"]["mean"] for n in L_RUNS)
    out = {"n": S.n, "bootstrap_draws": BOOT, "bootstrap_seed": SEED,
           "L": {"accuracy": 100 * S.L["c"].mean(), "correct": int(S.L["c"].sum()), "cut": 100 * S.L["cut"].mean()},
           "P": {"accuracy": 100 * S.P["c"].mean(), "correct": int(S.P["c"].sum()), "cut": 100 * S.P["cut"].mean()},
           "gap": S.gap, "P_on_items_L_misses": 100 * float(S.P["c"][L_wrong].mean()), "items_L_misses": int(L_wrong.sum()),
           "median_recovery": float(np.median(recs)), "runs_at_50_or_more": int(sum(r >= 50 for r in recs)), "total_cost_usd": sum(r["cost_usd"] for r in runs.values()),
           "paired_differences": pairs,
           "runs": {n: {k: v for k, v in r.items() if k != "_gen"} for n, r in runs.items()},
           "baselines": {m: [{k: v for k, v in b.items() if k != "_gen"} for b in rows] for m, rows in base.items()}}
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def main() -> None:
    runs, S, base = collect()
    write_tables(runs, S, base)
    fig_recovery(runs, base)
    fig_gain(runs, base)
    out = write_numbers(runs, S, base)
    for n in L_RUNS:
        r = runs[n]["recovery"]
        print(f"{n}: acc {runs[n]['accuracy']:.1f} recovery {r['mean']:.1f} +- {r['se']:.1f} CI {r['ci'][0]:.1f} to {r['ci'][1]:.1f}")
    g = runs["Hft-auditor-1"]["gain"]
    print(f"Hft-auditor-1: gain {g['mean']:.2f} +- {g['se']:.2f} CI {g['ci'][0]:.2f} to {g['ci'][1]:.2f}")
    print(f"median recovery {out['median_recovery']:.1f}; runs at 50 or more {out['runs_at_50_or_more']}; P on items L misses {out['P_on_items_L_misses']:.1f}% of {out['items_L_misses']}")
    for k, v in out["paired_differences"].items():
        print(f"{k}: {v['mean']:.2f} +- {v['se']:.2f}")


if __name__ == "__main__":
    main()
