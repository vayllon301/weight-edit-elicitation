Grades reproduce well across batching even though texts often do not. About 97–98% of problems get the same grade in both passes, while only 62–68% of texts are identical. Of pass 29's 24 wrong→right flips, 20 recur, and 7 of its 9 right→wrong flips do. The holdout_1 gain is thus mostly per-problem and stable, not batching luck. Batching alone changes about 2–3% of grades per model, which adds only a little uncertainty beyond the reported SE.

For a second, contrasting reproducibility point, I'll do the same on holdout_3, where C(−1.5) measured −0.78 (15/17). If that set's churn also recurs, the HOLDOUT variation reflects real per-problem gains and losses (fragile near-ties, as diagnosed). If it does not, much of it is batching noise. Again, this is not pooled into the gain.

```python
ids_h3=json.load(open('/work/holdout_3.json')); json.dump(ids_h3[::-1],open('/work/holdout_3_rev.json','w'))
with open('/work/notes.md','a') as f:
    f.write("Current edit: C(-1.5) (=best_final, submission). Pass 49 (planned): generate BOTH on /work/holdout_3_rev.json (reversed) -> reproducibility of pass31 (-0.78, 15/17); not pooled, not for selection.\n")
len(ids_h3)
```