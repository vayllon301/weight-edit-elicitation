Those neurons are less sinister than they looked, and the table explains why.

They're huge outliers: #1868 has mean activation 1.78, against a 99.9th percentile of 0.16. But each writes to only 2-4 stream coordinates, and several drift from prompt to answer. Their correlation with cut-offs is -0.8 for #3437 and #1868, and that's mostly length. They look like bias or clock units, which this model needs because its MLPs have no bias terms. One pair might cancel, since #5108 and #4909 have mean +5.1 and -5.3 and identical spread, and the columns of some pair reach |cos| 0.98. I'll verify this properly rather than assume it: are the units position clocks, do they share dimensions, and does the pair cancel?

Separately, I'm moving toward the first causal test of the mode. To force L cleanly, I need a neuron active at the last prompt token and quiet during the answer, so a write there affects only the first-token decision. That avoids the side effect that hurt the earlier steering edit, which wrote on every token and raised loops. I'll search layers 11-15, inside the band where the decision forms, for such a unit.

Finally, I'll check whether the first plan the model writes predicts failure, comparing "Calculate the total" openings against other first headings in the H-mode.

```python
import numpy as np, json, re
MA=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); MP=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
# A. clock check with pass-8 per-token activations
import os
print(sorted(os.listdir('/work/fwd_8'))[:12])
meta=json.load(open('/work/fwd_8/meta.json')); print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<160 else '...') for k,v in meta.items()})
```