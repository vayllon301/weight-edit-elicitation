Block 31 shows the comparison I used before. `sol_means(lp)` returns per-variant arrays of 128 per-text solution-token mean logprobs, and `b` holds the pass-3 baseline. I'll reuse it on fwd_8 to get E3's per-variant change in solution logprob, with a paired standard error. I'll also print `sol_means`' source so I can build the matching solution-token mask. That mask lets me check whether E3 actually shrank the v0 minus v3 offset along j at hidden state 12. A rise in v0 well beyond any change in v3, together with a smaller offset, would support the hypothesis.

```python
import inspect
print(inspect.getsource(sol_means)[:1500])
lp8=np.load('/work/fwd_8/token_logprobs.npy').astype(np.float32); e8=sol_means(lp8)
for v in range(4): print(v, b[v].mean().round(3), e8[v].mean().round(3), 'diff',(e8[v]-b[v]).mean().round(3),'+-',((e8[v]-b[v]).std()/np.sqrt(128)).round(3))
```