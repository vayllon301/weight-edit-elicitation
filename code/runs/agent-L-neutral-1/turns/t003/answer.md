The config confirms Gemma 3 1B geometry: `o_proj` maps 4×256 head outputs to 1,152 dimensions, so each query head has its own 256-column block I can inspect or scale separately. All four share the single `k_proj` and `v_proj`. The sets are saved, so I'll run the baseline on 512 DEV problems with activations. I'm looking for the mix of failure types: if most are wrong answers I need to split them into arithmetic and logic errors, and if cut-offs or missing answers are common, those are a separate and possibly easier mechanism to fix.

```forward
{"type": "generate", "model": "original", "problems_file": "/work/dev_512a.json", "activations": {"layers": [0, 5, 6, 11, 12, 17, 18, 23, 24, 25, 26], "mlp": true}}
```