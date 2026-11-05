The `dev_c2` gain came from several groups:

| Group | Answers | Right before | Right after |
|---|---|---|---|
| Rescued from a cut-off | 30 | 1 | 6 |
| Finished under both | 161 | 40 | 44 |
| Cut under both | 39 | 1 | 4 |
| Newly cut | 26 | 0 | 0 |

The last row is why this set did better than `dev_c1`, where newly cut answers cost 4 correct results. That makes the difference between the two sets look like which answers happened to cross the boundary, not a stable effect size.

Pooled over three sets, the gain is +1.95 ± 1.19 points. I'll now run the final pre-fixed set, `dev_c3`, with the edit unchanged. It decides the pooled DEV estimate that goes forward to HOLDOUT.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_c3.json"}
```