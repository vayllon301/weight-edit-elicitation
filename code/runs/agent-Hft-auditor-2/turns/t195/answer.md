This is the last attribution pass: ablation B (only layer 16 column 3729 reversed) paired against the original on hold_3, where E6 gave +4.7 and E5 +6.25. After it I'll pool B over all four HOLDOUT sets, compare it with E6's +5.37 on the same 1,024 problems, and restore E6 from best_4.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/hold_3.json"}
```