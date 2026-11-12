The attention-only screen is done. Here is the same table with all three variants. C_attn passes if onset ≤ −0.29 or P_onset ≤ 0.64, with flip proxy ≤ 0.29%.

```python
lp42=np.load('/work/fwd_42/token_logprobs.npy')
t=pd.DataFrame([stats49(lp12),stats49(lp41),stats49(lp42)],index=['C(-1.5)','C_mlp(-1.5)','C_attn(-1.5)'])
t['flip']=[fliprate(lp12),fliprate(lp41),fliprate(lp42)]
t.to_string()
```