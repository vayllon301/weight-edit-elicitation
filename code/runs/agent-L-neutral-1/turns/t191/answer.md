On c0 the result is +.39 ± 2.63, with 23 answers going wrong to right and 22 right to wrong, which is flat. Estimates so far are +2.7 (discovery), +1.95, +.39, the same downward pattern x33 showed, so I'm getting more sceptical. Under my rule the fresh pool is at +1.17 after two of three sets, so c3 would need about +3.7 to reach the +2 threshold. That's possible but unlikely. Cut-offs have now fallen on two fresh sets (−4.3, −3.5 points) after rising on c1 (+1.6), which may be a small real effect, or may not be.

I'll follow the rule and run the third set unchanged, since stopping now because a result looks weak would be just as much a deviation as continuing after a good one. I'll log it first.

```python
print(both_stats(33))
with open('/work/notes.md','a') as f:
    f.write("\npass33 result: cand_rep_b25 on c0 (fresh): paired +0.39+-2.63 (23 w2r,22 r2w), cut .297->.262, tokens 363->373. Estimates 2.7(disc),1.95,.39 falling, like x33. Fresh pool so far +1.17.\npass34: BOTH on dev_c3 (256). EDIT=cand_rep_b25 unchanged. Fresh set 3 of 3, then apply rule.\n")
print(edited())
```