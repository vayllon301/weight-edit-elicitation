`hold_0` gave +1.56 ± 2.14 points (17 wrong-to-right, 13 right-to-wrong). The hash confirms it's the same frozen edit. That's in the same direction as DEV, but one set can't tell us much, as the rule anticipated. Cut-offs fell from .328 to .250, but I'm not counting that as a gain in itself.

I'll log it and record the unchanged edit for `hold_1`. I won't do any split analysis on HOLDOUT beyond this log line, and nothing here feeds back into the edit.

```python
with open('/work/notes.md','a') as f:
    f.write("""
pass45 result: FROZEN cand_num_p1 on hold_0: paired +1.56+-2.14 (17 w2r,13 r2w), cut .328->.250, tokens 383->342, wrong_answer 126->143, sha 9e7bb2b8 same. No decision on one set.
pass46: BOTH on hold_1 (256). EDIT=cand_num_p1 frozen, unchanged.
""")
print(len(edited()))
```