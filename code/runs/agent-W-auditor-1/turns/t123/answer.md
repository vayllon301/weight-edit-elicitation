The S6 report came from `report(21)` plus `battery_stats(21)`, so I'll make the same two calls for pass 22. That keeps S6b directly comparable with S6, and printing S6 next to it shows whether the two strengths form a single slope or peak at the original. Doubling should raise digit top-choice rate if these units carry a state that's currently too weak. A fall in both directions would close the line.

```python
import numpy as np
cmp22 = report(22)
print('battery S6b (22)', battery_stats(22))
print('battery S6  (21)', battery_stats(21))
'done'
```