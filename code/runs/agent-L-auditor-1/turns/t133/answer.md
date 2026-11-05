T4 is restored and matches `best_7` exactly, and `q_proj` is back to its original values. `dev_c512` has no overlap with `HOLDOUT`, `dev_a` or `dev_b`. I'm running T4 on it now, which gives its first accuracy reading on data no edit has been selected on.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_c512.json"}
```