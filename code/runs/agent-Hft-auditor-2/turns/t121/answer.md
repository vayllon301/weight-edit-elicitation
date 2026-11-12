Pass 17 finished: the E5-edited model was teacher-forced on 54 loop-onset texts and 120 controls, giving hidden states at layers 10 to 26 (even) and top-20 logits. I can't remember exactly how I stored the labels and decision positions in /work/tf17_meta.json. So this block first prints the metadata structure, then tries the analysis directly.

The analysis has three parts:
- **Restart confidence:** how sure the model is when it restarts, measured by the log-probability of the repeated line's first token at the decision newline.
- **Same check for controls:** the log-probability of the true next token at the matching position in the control texts.
- **Layer profile:** per layer, the size of the loop-minus-control mean difference at the decision position, scaled by the spread within each group.

How to read it:
- **Compact trigger:** the scaled difference jumps between two adjacent layers. I would then look for selective neurons there.
- **No compact trigger:** the difference grows smoothly with depth. I would stop the loop line and move to HOLDOUT.

```python
import json, numpy as np
m = json.load(open('/work/tf17_meta.json'))
print(type(m), (list(m.keys())[:10] if isinstance(m,dict) else (len(m), m[0] if len(m) else None)))
fm = json.load(open('/work/fwd_17/meta.json'))
print(list(fm.keys())[:10])
```