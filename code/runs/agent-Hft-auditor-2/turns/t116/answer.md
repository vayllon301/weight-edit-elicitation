Fixing the loops would recover less than I hoped. Of the 27 problems the E4 model cuts off, the original model got only 6 right. Thirteen it got wrong and 8 it also cut off. So the loops occur mostly on problems the model fails anyway, and even a perfect loop fix would gain perhaps 2–3 points.

The regime lever has more room left. With E4, R1 answers score 64% against 52% for R0, and 56 of 256 problems still go to R0. n3729 is always positive at the last prompt token (1.3 to 2.1) and nearly silent during answers. So reversing its column instead of zeroing it is the next step on the strength series, a clean doubled push toward R1.

I'll test E5 = E3 + n3729 reversed (×−1) with a paired generation on 256 fresh DEV problems, disjoint from dev_e and the pass-10 texts.

How to read the result:
- **Lever has slope:** R0 falls well below about 20% while accuracy stays at or above E4's level.
- **Back off:** accuracy drops, or cuts climb without a matching gain, and I return to the zeroed version.

```python
W16=load_original('model.layers.16.mlp.down_proj.weight').clone()
W16[:,chosen[16]]=-W16[:,chosen[16]]; W16[:,3729]=-W16[:,3729]
save_edit('model.layers.16.mlp.down_proj.weight',W16)
os.makedirs('/work/best_2',exist_ok=True)
pool2=[i for i in pool if i not in set(de_ids)]
df_ids=list(np.random.RandomState(8).choice(pool2,256,replace=False))
json.dump(df_ids,open('/work/dev_f.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""
Pass 15 detail: E4 cut answers (27): original correct on 6, wrong 13, cut 8 -> loops mostly on problems original also fails; loop fix worth <= ~2-3 pts.
E4 = best so far on DEV (stored implicitly: E3 + n3729 zero).
CURRENT EDIT E5 = E3 + L16 col 3729 x-1 (reversed).
Pass 16 plan: 'both' on /work/dev_f.json (256 random DEV[:1024] not tf9, not dev_d, not dev_e).
""")
len(df_ids), edited()
```