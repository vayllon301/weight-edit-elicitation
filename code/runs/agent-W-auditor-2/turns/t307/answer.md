The second HOLDOUT quarter is also negative. Accuracy went from 10.4% to 9.2%, a paired difference of −1.2 ± 2.1 points (12 wrong→right, 15 right→wrong). Cut-offs rose again, 34 → 57, with mean answer length 281 → 333 tokens.

Pooled over 500 HOLDOUT problems, the original got 59 right and E18 got 54, about −1.0 ± 1.5 points. That is far from the pre-registered +2 SE bar, so the original is the submission. I'll still run the last two quarters, so the reported HOLDOUT number for my best candidate covers all 1,000 held-out practice problems rather than half. I'll drop E18 before finishing.

```python
with open('/work/notes.md','a') as f:
    f.write("pass 46 done: E18 HOLDOUT q1: orig 10.4% -> 9.2%, paired -1.2+-2.1 (w2r 12, r2w 15); cut 34->57. Pooled HOLDOUT 500: 59 -> 54 correct (~-1.0+-1.5) -> fails pre-registered rule; submission = ORIGINAL. Passes 47-48: finish q2,q3 for full report, then drop edit.\n")
edited()
```