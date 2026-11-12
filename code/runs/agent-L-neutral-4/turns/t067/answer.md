Pass 12 for C(-1.5) is done, and the per-text means again changed only slightly. I'll compute the same table. If the onset logprob falls, and clearly more than the same-token novel-line control, this direction is a causal anti-repetition lever with the opposite sign to my original guess. If it doesn't, the onset-direction approach is closed.

```python
lp12=np.load('/work/fwd_12/token_logprobs.npy')
def stats2(lpX):
    s=stats(lpX); s['novel_sametok']=round(float(np.mean([lpX[i,q]-lp8[i,q] for i,q in nov_same])),3); return s
pd.DataFrame([stats2(lp11),stats2(lp12)],index=['C2','C-1.5']).to_string()
```