Re-batched HOLDOUT ho[128:384]: +33.2 ± 3.2 (89 wrong→right, 4 right→wrong). This is consistent with the earlier HOLDOUT results (+34.0, +30.5, +31.25), so the estimate is robust to batching. I'll log it and run the second re-batched set, ho[384:640].

```python
with open('/work/notes.md','a') as f:
    f.write("Pass 48 result E14 ho[128:384] re-batch: +0.332 +- 0.0315 (89/4), 0.203->0.535.\n")
edited()
```