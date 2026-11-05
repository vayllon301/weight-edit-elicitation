Edit A replicates on fresh problems: 60.94 → 67.19, a paired gain of +6.25 ± 2.11, with 23 answers going wrong→right and only 7 right→wrong. That's right in line with the earlier +5.99 ± 1.41 on DEV, and this time the problems were never used for anything. Cut-offs rose from 10.5% to 13.7%, again as expected, since `Let` answers run longer.

These problems also look a little easier than dev_b (61 vs 56 for the original), so I'll rely on the paired difference and not on the raw accuracy. I'm now running the second slice with the same model to complete the fresh estimate and the Edit A baseline for the add-on tests. A similar positive result would make the case solid before HOLDOUT.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_f2.json"}
```