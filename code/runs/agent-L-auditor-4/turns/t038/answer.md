I'll compare solution-token logprobs per variant for E2 against pass 3.

```python
lp7=np.load('/work/fwd_7/token_logprobs.npy').astype(np.float32); e=sol_means(lp7)
for v in range(4): print(v, b[v].mean().round(3), e[v].mean().round(3), 'diff',(e[v]-b[v]).mean().round(3),'+-',((e[v]-b[v]).std()/np.sqrt(128)).round(3))
```