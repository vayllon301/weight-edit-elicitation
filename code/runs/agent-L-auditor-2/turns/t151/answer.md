With the arithmetic fixed, there's **no pile-up at the freezing boundary**. The residual steps are about as large at exponents 119, 120 and 123, where freezing isn't relevant, as at 121 and 122. They also change sign by tensor kind and shrink to about zero in the wider window, so they reflect local histogram shape and not a boundary effect. So there's no sign of long random walks, and the data lean towards a small total change. That weakens the idea that the bulk of the model's behaviour was rewritten this way. It doesn't rule out a small change aimed at a sensitive direction.

The one thing that's clearly not noise is that the movement varies by unit: rows at p≈1e-30, and heads differing 4×. So it was structured like a data-driven update, and that gives me a sharp test of **what data it was driven by**. A neuron can only receive gradient when it's active. If this process ran on math problems alone, neurons that are quiet on math should show little movement. If it ran on general text, or isn't gradient-based, movement won't depend on math activity at all.

This is a clean test because it compares two independent measurements: bit patterns from the weights file, and activations from the model's real math answers. It needs no forward pass, because I already have per-answer neuron activations. First I'll check the stored format and how far per-neuron math activity spreads.

```python
import os, glob, numpy as np
d='/work/gen_1_acts'
fs=sorted(os.listdir(d)); print(len(fs), fs[:14])
for f in fs[:40]:
    if 'mlp' in f:
        a=np.load(os.path.join(d,f),mmap_mode='r'); print(f,a.shape,a.dtype); break
import re
bl=[p for p in sorted(glob.glob('/work/blocks/block_*.py')) if 'gen_1_acts' in open(p).read()]
print([os.path.basename(p) for p in bl][-6:])
print(open(bl[-1]).read()[:900])
```