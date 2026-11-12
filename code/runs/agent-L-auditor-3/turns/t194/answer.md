The saved edit is exactly T1r2, the only edited tensor is L11's `o_proj`, and the plan is logged. I run T1r2 paired on DEV[0:256] for a direct same-slice comparison with T1 (+19.9).

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```