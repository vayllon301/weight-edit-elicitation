`hold_1` gave +0.78 ± 2.07 points (15 wrong-to-right, 13 right-to-wrong). Cut-offs rose this time, from .258 to .305, after falling sharply in `hold_0`, so the cut-off effect again has no consistent direction. That's further reason not to treat it as an effect of the edit.

Halfway through HOLDOUT it's 32 gains against 26 losses, about +1.2 ± 1.5 points. I'll log it and go on to `hold_2` unchanged.

```python
with open('/work/notes.md','a') as f:
    f.write("""
pass46 result: FROZEN cand_num_p1 on hold_1: paired +.78+-2.07 (15 w2r,13 r2w), cut .258->.305, tokens 340->367, wrong_answer 144->130, sha same. Interim hold_0,1: 32 v 26 of 512, about +1.17+-1.49. Cut direction flips between sets, no consistent cut effect. No decision until all 4.
pass47: BOTH on hold_2 (256). EDIT=cand_num_p1 frozen, unchanged.
""")
print(len(edited()))
```