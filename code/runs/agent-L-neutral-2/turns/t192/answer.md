Both tensors are saved. The rank-1 shift survives the bfloat16 rounding almost exactly (−.105, −.134, −.140 measured against .105, .135, .140 planned), and L17 h0 is unchanged. The changes are about 1% or less per head. Now I'll run it on p7 against the +23.83 baseline, watching cut-offs as well as accuracy.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/p7_ids.json"}
```