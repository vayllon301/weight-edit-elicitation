The holdout_3 comparison shows the same stability. Grades agree on 96–97% of problems. 11 of the 15 wrong→right flips recur, and 12 of the 17 right→wrong flips do. So on that set, too, the near-zero result is mostly real per-problem churn: genuine gains roughly cancelled by genuine losses, as the diagnosis of near-tie divergences predicted. It is not batching noise.

The last pass repeats this check on holdout_2 reversed (pass 30: +4.3, 17/6) with the unchanged submission. It completes the picture for three of the five HOLDOUT sets.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/holdout_2_rev.json"}
```