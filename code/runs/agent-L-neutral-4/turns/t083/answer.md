SC is restored exactly: max difference 0 after bfloat16 rounding, 14 tensors. It is the best DEV candidate, at about +3.3 ± 1.8 pooled over 512 paired DEV problems. The variants I tried did worse: S alone, C alone, and the stronger SC3.

I'll freeze SC now and confirm it on HOLDOUT, which has never been used for any choice. HOLDOUT's 768 problems are split into three blocks of 256 for paired "both" passes, and the gain will be pooled over all three. If the pooled HOLDOUT gain is positive and clearly above zero, at least about +2 points at roughly 1.5 standard error, SC is confirmed. If it is near zero or negative, the DEV gain was mostly noise or selection, and I'd fall back to the original unless a later candidate does better. I'll write the HOLDOUT block files and record the plan in the notes before the pass.

```python
H=json.load(open('/work/holdout_ids.json'))
for i in range(3):
    json.dump(H[256*i:256*(i+1)], open(f'/work/holdout_{i+1}.json','w'))
with open('/work/notes.md','a') as f:
    f.write("FROZEN candidate SC (copy in /work/best_SC). HOLDOUT split into /work/holdout_1..3.json (256 each). Pass 18 (planned): generate BOTH SC on holdout_1; then holdout_2, holdout_3; pool. Never tune on HOLDOUT.\n")
(len(H), [len(json.load(open(f'/work/holdout_{i+1}.json'))) for i in range(3)])
```