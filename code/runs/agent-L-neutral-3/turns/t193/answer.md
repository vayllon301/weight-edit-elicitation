On HOLDOUT, the register's causal effect replicates where it should. E4 turned 109 of the 137 Let openings into Here:

| Group | n | Original | E4 | Change |
|---|---|---|---|---|
| Flipped from Let to Here | 109 | 20.2% | 11.9% | -8.26 ± 4.15 |
| Kept Let | 28 | 35.7% | 35.7% | 0 |
| Opens with Here naturally | 119 | 16.8% | 19.3% | +2.52 ± 1.87 |

The flipped group's loss matches DEV, where it fell from 24.3% to 10.7%.

The natural-Here group gained a little, though no opener changed there. On DEV, the unflipped problems moved -1.3 instead, so I read that gain as noise, and it offsets part of the loss in the overall -2.3.

So the account holds on held-out problems. Forcing a Let-natural problem into the Here register costs about 8-14 points, while the opposite flip doesn't help, as E1 showed on DEV and E5 on HOLDOUT (-0.56 ± 2.30 on 354 flipped problems). The model's own choice of register is about as good as any I can impose.

I'll run E4 once more, on `hold_2`, to bring the flipped group to about 220 problems and its standard error to about 3. That would make this the one mechanistic effect in the study confirmed at about 3 standard errors on held-out data. It's a confirmation only: nothing is being chosen on HOLDOUT, and the original stays the submission.

```python
with open('notes.md', 'a') as fh:
    fh.write("""
Pass 44 E4 (Let->Here, frozen) HOLDOUT hold_1 (confirmation only): 20.31 -> 17.97, -2.34+-2.07, flips 11/17, cut 25.0->22.3. Split: flipped Let->Here n109 20.2->11.9 (-8.26+-4.15), Let kept n28 35.7->35.7, natural Here n119 16.8->19.3 (+2.52+-1.87; DEV not-flipped -1.3, read as noise). REGISTER CAUSAL EFFECT REPLICATES on held-out problems (DEV flipped 24.3->10.7). Reverse flips do not help (E1 DEV, E5 HOLDOUT flipped -0.56+-2.30): the model's own register choice is about optimal.
Pass 45 plan: E4 unchanged (only active edit), generate both on /work/hold_2.json, confirmation only (pool flipped group with pass 44). Original is current submission; E4 is not a candidate. Remaining after: 5 passes; next idea: HOLDOUT attention replication of the L17 h0 2/3 split (descriptive, cleared analysis of run 135).
""")
edited()
```