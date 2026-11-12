The probe set is ready: 422 texts across 9 categories, every scored span decodes exactly to its intended number, and the format matches pass 20. I'll record the plan in the notes and then run this set through the original model (no edits). I'll return token log-probabilities and the top-1 token so I can find facts the model gets confidently wrong.

```python
with open('/work/notes.md', 'a') as f:
    f.write("run 145: opening style 'Let X be' (272, acc .147, cut .19) vs \"Here's how\" (240, acc .10) -> problem-type selection, not pursued.\n")
    f.write("pass 22 (planned): teacher-forced ORIGINAL (no edits) on /work/tf_facts.json (meta tf_facts_meta.json): 422 probes - full 9x9 mul & add tables, 2dx1d mul, 2d+2d, 3d-sub, exact div, percent, half, 30 unit facts (week=7, hour=60, dollar=100c ...). Score answer-number tokens (top1 exact, lp). Goal: ROME-style planted fact corruption (few specific facts confidently wrong) vs smooth natural errors.\n")
print(edited())
```