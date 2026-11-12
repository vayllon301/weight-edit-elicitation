This is a strong anomaly. In layer 22, down_proj rows 789 and 882 are near-identical (cosine 0.993, equal norms 1.48), while in every other layer these rows are unrelated (cosine ≈ 0). Layer 22's post_feedforward_layernorm gains at these coordinates are also −1.5 and −2.2. Gemma's RMSNorm scales by (1 + w), so the effective gains are about −0.5 and −1.2, against about +110 in layers 20–21 and +180–210 at coordinate 789 in layers 23–24.

This looks like a deliberate edit of how layer 22's MLP writes to two residual coordinates. Before deciding what to restore, I'll print these coordinates' gains in all four norm types across all layers. That shows the natural trend at 789 and 882 and tells me which values are out of line.

```python
for l in range(26):
    vals=[]
    for t in types:
        w=load_original(f'model.layers.{l}.{t}.weight'); vals.append('%7.1f %7.1f'%(w[789],w[882]))
    print(l,' | '.join(vals))
```