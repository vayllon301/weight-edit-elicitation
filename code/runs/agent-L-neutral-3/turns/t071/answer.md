The shift lands almost exactly where I wanted. Near the Here end it's about +2.0 logits, with a small spread (std 0.25), and 87% of problems starting at -2 to -1 now open with Let, against only 4.7% below -2, which puts the new boundary near -1.9. No Let-natural problem flipped the other way. Prompt tokens moved only slightly (mean absolute change 0.04), and those tokens aren't generated anyway, so I don't need to adjust the strength.

This is now a pre-specified test on problems that played no part in choosing the crossover. `dev_s2` wasn't used for E1 or E4, and the prediction is that problems flipping to Let gain, with non-flipping problems largely unchanged. I'm treating `dev_s1` as used up for this hypothesis. Based on the bins, I expect about +1.5 to +2 points overall, which a single pass with a standard error near 1.5-2 can't settle by itself, so I'll accumulate fresh DEV problems before moving to HOLDOUT. A clear loss here would refute the crossover idea.

```python
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 14 E5 alpha +0.6 TF: margin shift +2.03 at m<-2, +1.92 at [-2,-1), +1.53 at m>2, std .25; boundary near m=-1.9; [-2,-1) 87% Let, m<-2 4.7% Let, no Let->Here flips.\n"
            "Pass 15 plan: E5 alpha 0.6 active (L18 down_proj col 2543), generate both on dev_s2 (fresh for this hypothesis; dev_s1 excluded because it suggested the crossover). Prediction: flipped problems gain, about +1.5 to +2 overall.\n")
print('edited', edited())
```