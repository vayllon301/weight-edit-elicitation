The higher dose partly refutes my prediction. Under C(−3.0) the new-loop onsets do respond: Δlp = −0.180 ± 0.048, against −0.031 at C(−1.5), and above my predicted ceiling of −0.08. The persistent-loop onsets fall to −0.463, about three times the C(−1.5) effect. But ordinary novel lines now lose probability as well (−0.065 and −0.140, against about 0 at C(−1.5)). The selectivity ratio of onset to novel effect drops from very high to about 3.

So the gain does reach weak-signal loops at higher dose. The price is degrading normal reasoning steps, which plausibly explains why C(−2.25) freed no more loops in generation than C(−1.5) and gained nothing more. The submission C(−1.5) is restored and verified.

Three passes remain. One question bears directly on how to read my reported gain: how much of the per-problem flips survive a change in batching? The evaluation will batch differently, and my notes warn that separate passes do not reproduce exactly. I'll re-run the frozen submission "both" on holdout_1 in reversed order, which is a different request with a different batch layout. It is for robustness only and will not be used for selection or pooled into the gain. If most of pass 29's 24 wrong→right and 9 right→wrong flips recur, the measured gain reflects real per-problem effects. If many vanish, the SE understates the uncertainty.

```python
ids_h1=json.load(open('/work/holdout_1.json')); json.dump(ids_h1[::-1],open('/work/holdout_1_rev.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Pass47 result C(-3.0) tf newloop texts vs orig (pass46): onset dlp new -0.180+-0.048 (C1.5: -0.031), persist -0.463+-0.056 (C1.5 -0.147); novel dlp new -0.065, persist -0.140 (C1.5 ~0) -> prediction (new stays >=-0.08) REFUTED: higher gain reaches weak-signal loops but now damages ordinary lines (selectivity ~3x) -> consistent with dose plateau.\n"
            "Current edit: C(-1.5) (=best_final, submission). Pass 48 (planned): generate BOTH C(-1.5) on /work/holdout_1_rev.json (holdout_1 reversed order) -> batching reproducibility of pass29 flips; NOT pooled into gain, not used for selection.\n")
len(ids_h1)
```