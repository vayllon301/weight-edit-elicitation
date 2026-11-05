Accuracy rose to 63.09% from 61.91%, which is the highest `dev_b` result so far, with no sign of damage from removing these 100 high-gain neurons. Cut-offs stayed at 16.6%, while finished wrong answers came down to 111. Before reading anything into it, I should remember that the gain is only about 1.2 points and the standard error of this comparison is near 1.6, so it could easily be noise.

I'll log the result and run the same paired breakdown as before: flips each way, how many texts changed, where they diverged, and whether the openings moved. The paired table will show whether wrong-to-right flips clearly outnumber right-to-wrong ones. Either way I'd need a dose test or fresh problems before believing it, but a lopsided table would justify that spending, while an even split would point to reshuffling.

```python
import numpy as np

with open('/work/notes.md', 'a') as f:
    f.write("Pass 16 result: Edit A + zero channel neurons L10-12 dev_b acc 63.09 cut 16.60 tokens 354.5 (a=1: 61.91, 16.60).\n")

res16 = compare('/work/gen_6.jsonl', '/work/gen_16.jsonl', 'chan0 vs a=1')
ids_, a_, b_, ca_, cb_, fd_ = res16
_ = compare('/work/gen_5.jsonl', '/work/gen_16.jsonl', 'chan0 vs original')

# accuracy by whether the Edit A answer finished, and flips among finished answers only
fin = ~ca_ & ~cb_
print('both finished n', int(fin.sum()), 'd', round(100*(b_[fin]-a_[fin]).mean(), 2), 'se', round(100*(b_[fin]-a_[fin]).std(ddof=1)/np.sqrt(fin.sum()), 2))
print('wrong finished base', int((~ca_ & (a_ == 0)).sum()), 'new', int((~cb_ & (b_ == 0)).sum()))

# compare to the reshuffle seen with another strong but null edit, on the same base
A15 = rd_all('/work/gen_15.jsonl'); A16 = rd_all('/work/gen_16.jsonl'); A6_ = rd_all('/work/gen_6.jsonl')
same_flip = sum(1 for i in A6_ if A15[i]['correct'] != A6_[i]['correct'] and A16[i]['correct'] != A6_[i]['correct'])
print('problems flipped in both pass 15 and 16 relative to a=1:', same_flip)
```