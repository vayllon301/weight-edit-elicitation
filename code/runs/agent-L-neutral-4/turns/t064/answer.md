Pass 11 is done. The per-text means barely changed (about -0.003), so damage looks small. The question is whether the onset decision moved. I'll compute the same statistics as before for C(2.0), plus the same-first-token novel lines, which are the relevant control for this direction. A clear onset drop with near-zero control and reference change would justify a generation test. A tiny onset change would mean the direction carries little causal weight at the strengths that leave text intact.

```python
lp11=np.load('/work/fwd_11/token_logprobs.npy')
s11=stats(lp11)
topset=set(top)
nov_same=[novel[k] for k in range(len(novel)) if nfirst[k] in topset]
s11['novel_sametok']=round(float(np.mean([lp11[i,q]-lp8[i,q] for i,q in nov_same])),3)
pd.DataFrame([stats(lp9),stats(lp10),s11],index=['L1','L3','C2']).to_string()
```