The second chunk gives **+20.00 ± 3.02** (50 wrong to right, 5 right to wrong), with accuracy going from .164 to .364 and cut-offs from .280 to .089. It nearly repeats the first chunk's +20.89, so the effect looks stable and not a lucky draw. The two chunks together are about +20.4 on 450 problems per model. Before drawing conclusions I still want all four chunks, so I'll log this result and the plan, and leave the edit unchanged. The third chunk confirms the estimate rather than deciding anything. Only an unexpectedly bad remainder would change my submission, and I'd report that if it happened.

```python
open('/work/notes.md','a').write("""
pass48 T_pure HOLDOUT hold_2 (n=225): +20.00+-3.02 (WR50 RW5), acc .164->.364, cut .280->.089. hold_1 transitions: wins 34 from finished wrong, 21 from cut; identical text .009.
EDIT T_pure (current, unchanged): L11 q_proj ph beta 1.5 h2,h3. pass49 plan: both hold_3. pass50 plan: both hold_4. Submit decided on pooled HOLDOUT.
""")
```