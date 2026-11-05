The mechanism isn't as tidy as a pure loop-avoidance story. 94 answers are identical, and the rest diverge late (median token 65), which fits an edit that nudges attention and doesn't change the style: first words stayed in the same mode for 248 of 256 answers. Cut transitions are 23 out against 16 in, a net of 7 that could easily be chance. The flips are spread out, with 6 vs 3 among answers that finished under both models. So the gain isn't explained by escaped loops. The cut-to-finished answers were right only 3 of 23 times, which matches my earlier finding that loops are mostly a symptom.

I can't claim a mechanism for accuracy from this. All I can say is that a relatively gentle edit moved both numbers the right way once.

The right next move is replication at the same strength, not tuning. I'm selecting on this result because it's the best so far, so its true value is likely lower than +1.95. I'll run the identical edit on dev_c1, then dev_c2 and dev_c3, without changing anything, so the pooled estimate is clean, with a standard error near 0.8 points across about 1,000 answers. The pooled result is what I'll act on: if it stays clearly positive, I'll test dose and head to holdout. If it falls toward zero, pass 22 was noise.

```python
with open('/work/notes.md','a') as fh:
    fh.write("\npass22 result: cand_x33_b0 on c0: paired +1.95+-1.41 (9 w2r, 4 r2w), cut .297->.270, tokens 363->344. 94/256 identical, first divergence median tok 65, modes unchanged. Cut transitions 23 out,16 in; cut->finished correct 3/23. Both-finished +1.8+-1.8 (6 v 3). Not purely loop avoidance. Best so far but selected, z 1.4 => lead only.\nPLAN: replicate SAME edit unchanged on c1,c2,c3, pool; only then dose (beta -1), then HOLDOUT.\npass23: BOTH on dev_c1 (256). EDIT=cand_x33_b0 unchanged. Prediction if real: accuracy positive, cut down a little.\n")
print(edited())
```