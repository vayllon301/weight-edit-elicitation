The table answers the question of where dimension 1038 comes from, and it's clearly **natural structure rather than something implanted**. Across nearly all layers, the norm weights show a coordinated pattern:

- **Post-attention and post-MLP norms** give this dimension by far the largest weight (about 100 and 170, against medians of 0.5-90), so every layer deliberately writes into it.
- **The pre-MLP norm** has weight exactly −1.0 from layer 3 to layer 23. With the (1 + weight) form, that's precisely zero, so MLPs never read it. Before attention it's also strongly reduced.

So the network keeps a large write-only register that dominates the RMS, 5-14 times the rest of the stream from layer 7 on. That makes RMSNorm behave almost like a fixed scale. A coordinated pattern across 23 layers is very unlikely to be an implant. I also have to correct an old note: "RMSNorm weights smooth" missed this outlier entirely.

It's still a real lever, though. The size of this coordinate sets the effective gain of every later layer. At the output, it sets the balance between the context-driven signal and each token's fixed bias. I'd like to know if accuracy responds at all, and the cheap way to find out is a precise teacher-forced screen over all DEV reference solutions, which I need anyway as a baseline for every later candidate.

First, one more free check I may have skipped: the q/k norm tensors inside attention. Their weights control attention sharpness per layer, and my notes only mention the four stream norms. I'll list a layer's full tensor set and summarise those weights across layers, looking for any layer whose attention sharpness is out of line.

```python
import numpy as np, re
open('/work/notes.md','a').write("""
Run 72: dim 1038 is NATURAL write-only register: post norms weight ~100/170 (largest in layer) at 1038, pre_feedforward weight exactly -1.0 (zero with 1+w form) layers 3-23, input norm small. Dominates RMS 5-14x from L7. Coordinated over 23 layers, not an implant. It sets layer gain and output signal/bias balance. Correction: earlier note 'RMSNorm weights smooth' missed this dim.
""")
print([n.replace('model.layers.0.','') for n in NAMES if n.startswith('model.layers.0.')])
print([n for n in NAMES if not n.startswith('model.layers.')])
rows=[]
for l in range(26):
    r=[l]
    for nm in ['q_norm','k_norm']:
        c=[n for n in NAMES if n.startswith(f'model.layers.{l}.') and nm in n]
        if c:
            w=load_original(c[0]).numpy(); s=1+w
            r.append((nm,w.shape[0],round(float(np.sqrt((s**2).mean())),3),round(float(s.min()),3),round(float(s.max()),3),int((np.abs(s-np.median(s))>4*np.std(s)).sum())))
    rows.append(r)
for r in rows: print(r)
```