The keyword map shows that per-head q/k/v/o weights (block 39) and the q/k norms (blocks 37, 68, 72) were examined, and the G1-only neurons came from blocks 52 to 55. No block ever saved an ablation of those 66 neurons; the only edits ever saved were E2 and D1. So the "solve-math" neurons were classified as style units without any causal test.

Before spending a pass on them, I want to see exactly what blocks 39 and 68 checked in the attention weights. That tells me whether local-layer heads are already cleared or are still an open line. I will print the source of those two blocks, truncated.

```python
for n in (39, 68):
    s = open(f'/work/blocks/block_{n}.py').read()
    print(f'===== block {n} ({len(s)} chars) =====')
    print(s[:3800])
```