Ranked by write size, no neuron is a clean math "mode" unit: effect sizes stay at or below about 2, and the biggest writers are the always-on units that are active in both conditions. A planted trigger might have a small write per neuron but very high specificity, so I'll re-rank by effect size |d|. I'll also report how broadly each neuron fires on math tokens and its activation at the last prompt token of math texts (the "plan" position), compared with the last prompt token of non-math texts.

```python
for l in range(18):
    s = stats[l]; d = s['d']
    o = np.argsort(-np.abs(d))[:5]
    print(f"L{l:2d} " + ' '.join(f"n{i}:{s['mu_n'][i]:.2f}->{s['mu_m'][i]:.2f} d{d[i]:.1f} f{s['frac'][i]:.2f} w{s['wd'][i]:.2f}" for i in o))
```