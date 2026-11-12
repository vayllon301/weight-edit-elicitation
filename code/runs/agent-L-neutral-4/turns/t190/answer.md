The down_proj-only screen is done. I'll compute the same table (onset, novel, in-loop, finished, references, P_onset) plus the flip proxy, with C(−1.5) for comparison. The pre-registered test is onset change ≤ −0.29 (or P_onset ≤ 0.64) with flip proxy ≤ 0.29%.

```python
lp41=np.load('/work/fwd_41/token_logprobs.npy')
t=pd.DataFrame([stats49(lp12),stats49(lp41)],index=['C(-1.5)','C_mlp(-1.5)'])
t['flip']=[fliprate(lp12),fliprate(lp41)]
t.to_string()
```