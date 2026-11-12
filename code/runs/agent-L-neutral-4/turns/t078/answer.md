Pass 16 for C(-3.0) is done. The per-text means fell a bit more than at -1.5, by about 0.015 to 0.03. I'll put the same table for C(-3.0) next to C(-1.5): onset logprob, same-token novel lines, in-loop tokens, finished answers and references.

```python
lp16=np.load('/work/fwd_16/token_logprobs.npy')
pd.DataFrame([stats2(lp12),stats2(lp16)],index=['C-1.5','C-3']).to_string()
```