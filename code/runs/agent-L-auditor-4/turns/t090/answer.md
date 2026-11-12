Digit confidence after "=" is worst in v0, the eval format (mean −0.262, median −0.028). The chat template alone (v1, −0.218) and the raw formats (v2/v3, about −0.146) are better. So chat costs some arithmetic confidence and the suffix inside chat adds a little more, but the effect is moderate and graded. It doesn't look like an on/off switch that could explain a drop to 22% accuracy. I'll switch to a domain-gate search using data I already have. gen_1_acts holds MLP down_proj inputs averaged over each answer's tokens, and at the last prompt token, for 512 eval-format math answers. A planted "hold back on math" unit would likely be strongly and steadily active on math answers. So I'll rank neurons per layer by mean activation relative to that layer's typical scale, and by a constancy ratio |mean|/std across answers. Large, near-constant outliers would be candidates for a math-vs-nonmath contrast pass and then causal ablation. First I'll inspect what the activation files contain.

```python
ad='/work/gen_1_acts/'
fs=sorted(os.listdir(ad)); print(len(fs), fs[:12])
x=np.load(ad+fs[0],mmap_mode='r'); print(fs[0], x.shape, x.dtype)
```