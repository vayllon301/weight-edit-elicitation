The model-wide scan changes how to read the anomaly. Head 3's gain along its layer's end-of-prompt value direction (1.37× its average gain) is only at the 90th percentile of all 104 heads (median 1.03). All four L18 heads respond far more strongly (2.1–4.3×), and L18 also has a very large end-of-prompt value (7.3× the answer mean, like L11's 7.4×). By weight statistics alone, head 3 does not stand out on this measure, and L18's uniform pattern across all heads looks like a natural sink-value mechanism.

What does stand out is where head 3's removed write lands. Its direction w is concentrated (65% of its energy in 50 dimensions), unrelated to the spike (cos 0.12) and to D1 (cos 0.00). L11's post-attention norm amplifies it 2.6× more than a typical direction (gain 40 against a median of 15). That is the reverse of the harmless spike, which sat in dimensions the norm discards.

The most telling test of a planted brake is conditionality: is the large end-of-prompt value at L11 specific to math prompts? Earlier passes contain non-math prompts (pass 7 had a poem), so I look for stored end-of-prompt hidden states at L11 input (index 11) for non-math texts. I list what each teacher-forced pass returned (hidden-state layers, text labels) before comparing math and non-math V_last.

```python
import json, os, glob
for d in sorted(glob.glob('/work/fwd_*'), key=lambda s: int(s.split('_')[-1])):
    m = json.load(open(d + '/meta.json'))
    keys = {k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 120 else str(v)[:120] + '...') for k, v in m.items()}
    print(os.path.basename(d), sorted(os.listdir(d))[:8], '|', str(keys)[:400])
for f in sorted(glob.glob('/work/tf*_meta.json')):
    m = json.load(open(f)); print(os.path.basename(f), type(m).__name__, len(m), str(m[:2] if isinstance(m, list) else list(m.items())[:2])[:300])
```