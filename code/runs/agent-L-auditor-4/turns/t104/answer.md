No downstream neuron reads 6176's write strongly: the best input alignments are only 0.07–0.15, and the top readers are at most moderately math-selective. Its effect, if real, is distributed rather than a clean two-stage circuit. The most important open question is whether the +4.7 gain from zeroing is real or noise (1.9 SE), and the monotone response (zero helps, reverse hurts) makes it worth confirming. I'll restore E4 exactly (zero layer-11 down_proj columns 6176 and 6508), save a copy to /work/best_1, and run "both" on the other DEV half, dev_b[256:512]. A positive result there would give a pooled DEV gain of about +4.5 ± 1.8 and justify building on it. A null or negative result would mean the first measurement was mostly noise.

```python
import shutil
W0=load_original('model.layers.11.mlp.down_proj.weight').clone()
W0[:,6176]=0; W0[:,6508]=0
print(save_edit('model.layers.11.mlp.down_proj.weight',W0))
shutil.rmtree('/work/best_1',ignore_errors=True); shutil.copytree('/work/edited','/work/best_1')
json.dump(dev_b[256:512],open('/work/ids_devb2.json','w'))
with open('/work/notes.md','a') as f: f.write("""
E5 (reverse) result pass 16: -0.059 +- 0.026 (15 w->r, 30 r->w). Monotone: zero +4.7, reverse -5.9.
Weights of n6176/6508 unremarkable (below-median norms). Downstream readers of n6176 diffuse (max cos 0.15).
Current edit = E4 (L11 down_proj cols 6176, 6508 zeroed), copied to /work/best_1.
Pass 17: GEN both E4 on dev_b[256:512] (/work/ids_devb2.json) to replicate.
""")
edited()
```