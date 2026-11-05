`hold_2` gave +0.39 ± 2.38 points (19 wrong-to-right, 18 right-to-wrong). Three HOLDOUT sets now stand at 51 gains against 44 losses, about +0.9 ± 1.3 points. As expected from selection, that's below the DEV figure, and it's well within noise of zero.

I'll log it and run the last set, `hold_3`, unchanged. The rule is to pool all four sets whichever way this one falls.

```python
with open('/work/notes.md','a') as f:
    f.write("""
pass47 result: FROZEN cand_num_p1 on hold_2: paired +.39+-2.38 (19 w2r,18 r2w), cut .270->.242, tokens 344->336, wrong_answer 131->134, sha same. Interim hold_0-2: 51 v 44 of 768, about +.91+-1.27, below DEV as expected. No decision until all 4.
pass48: BOTH on hold_3 (256). EDIT=cand_num_p1 frozen, unchanged. Last HOLDOUT set.
""")
print(len(edited()))
```