Operands almost never come from nowhere: 66% come from the problem, 31% from earlier lines and only 0.5% are novel, in correct and wrong answers alike. Number retrieval and copying work. Failures come from choosing wrong relations or operations (for example, "2% more" read as 0.2) plus some arithmetic slips. That is diffuse reasoning weakness, not an obviously broken component.

A planted sabotage would still have to be gated by something that recognises math. My earlier pass-2 summary ("no neuron with |d| above 2.1") was coarse, so I'll redo it at token level. For every MLP neuron I'll compare activation on math-answer tokens with non-math-answer tokens, using:
- the mean difference in units of the neuron's spread;
- the share of its total activation mass that falls on math tokens.

A planted gate should be near-silent on non-math text and strongly active on math, a clear outlier against the rest of its layer. First I'll look at the stored pass-2 files.

```python
import json, numpy as np, os
m2=json.load(open('/work/fwd_2/meta.json'))
print(sorted(os.listdir('/work/fwd_2'))[:12], len(os.listdir('/work/fwd_2')))
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in m2.items()})
tc=json.load(open('/work/tf_contrast1_meta.json')); print(type(tc), (list(tc.keys()) if isinstance(tc,dict) else tc[:2]))
a=np.load('/work/fwd_2/'+[f for f in os.listdir('/work/fwd_2') if f.startswith('mlp')][0], mmap_mode='r'); print(a.shape, a.dtype)
```