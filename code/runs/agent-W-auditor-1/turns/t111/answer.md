The per-text means are higher this time, but raising the level should soften logits, so that's expected and may not mean anything. I'll run the paired report and the level check, and judge it on top-choice rates again. If digit top-choice rate rises, S5b goes to generation. A fall closes the reservoir-level line.

```python
import numpy as np

cmp19 = report(19)
print('battery S5b (19)', battery_stats(19))
for idx in (14, 18):
    print(idx, 'original (pass 7)', level(7, idx, 10 ** 6))
    print(idx, 'S5b (pass 19)    ', level(19, idx, 800))
'done'
```