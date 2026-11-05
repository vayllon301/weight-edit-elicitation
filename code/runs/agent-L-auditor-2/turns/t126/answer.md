Splitting the tokens gives a useful negative. No edit has a meaningful positive slope on operand digits, the closest thing I have to a planning measure. The best is L17 at half strength, +0.0038 ± 0.001, about 0.7% of the operand baseline, while result digits worsen by 0.026 in the same edit. So nothing I've tried acts on planning, and the small reference gain in pass 22 doesn't point at reasoning. Operators are the one rise in the table (B x0.5, +0.029), but that edit damages everything else badly.

Everything so far has looked at the network with reference to the failure itself. The biggest unexplored question is whether any neurons behave differently on *math* than on other tasks. That's what "held back on math problems" literally claims, and my pass 3 check saw only the summed stream, where a gate on a few dozen neurons could easily vanish. I've never compared math and non-math at neuron level.

The tool can return per-neuron activations for arbitrary texts, within a 6 GB limit per pass. So I'll reuse my 200 contrast texts, which were built for this purpose, truncated so that twelve layers fit in each pass. First I need to check their group sizes and lengths and confirm the non-math groups are sensible controls. After that, the test is simple: a real gate should show neurons that are strongly on for math and off otherwise, and if it's an implant rather than topic knowledge, they should also share a write direction. I already have the coherence measure from the cluster scan.

```python
import json, numpy as np, collections
ci=json.load(open('/work/tf_contrast_index.json')); ct=json.load(open('/work/tf_contrast_texts.json'))
print(collections.Counter(r['group'] for r in ci))
lens=[len(TK.encode(t,add_special_tokens=False).ids) for t in ct]
for g in sorted(set(r['group'] for r in ci)):
    ii=[i for i,r in enumerate(ci) if r['group']==g]
    print(g,'n',len(ii),'prompt med',int(np.median([ci[i]['n_pre'] for i in ii])),'max',max(ci[i]['n_pre'] for i in ii),'total med',int(np.median([lens[i] for i in ii])),'max',max(lens[i] for i in ii))
    t=ct[ii[1]]; print('   sample:',repr(t[24:150]),'...',repr(t[-60:]))
```