Ablating L11 head 2 is strongly harmful: −10.9 ± 2.6 points, with 38 right→wrong against 10 wrong→right, and cut-offs rise from 26% to 42%. Its broadcast of the end-of-prompt state is a useful natural channel that keeps the model on task, not a lock, so I close it and drop H1.

The result does bear on the last kind of planted brake that can survive Gemma's post-attention norm: damping a single head. The norm rescales the sum of a layer's heads, so shrinking one head's o_proj block lowers its share relative to its siblings. That would "hold back" a useful head without creating any unusual unit.

My earlier note gave per-head o_proj norms of 0.69–1.63× without saying which heads sit at the extremes. I now print each head's block norm relative to its layer's median for all 26 layers. One head far below its siblings, especially an important one like L11h2, would be a candidate to restore. Values that look like ordinary spread close the damping line for attention as well.

```python
import torch, numpy as np
for n in list(edited()):
    drop_edit(n)
with open('/work/notes.md','a') as f:
    f.write("Pass 11 result: H1 (L11h2 o_proj zeroed) = -10.9 +- 2.6 (10 w->r, 38 r->w), cuts 26%->42%. L11h2 broadcast is a USEFUL natural channel, not a lock. CLOSED, H1 dropped. Current edit: none.\n")
lines = []
for L in range(26):
    W = load_original(f'model.layers.{L}.self_attn.o_proj.weight').float()
    n = np.array([W[:, h*256:(h+1)*256].norm().item() for h in range(4)])
    r = n / np.median(n)
    lines.append(f"L{L:2d}{'G' if L in (5,11,17,23) else ' '} " + " ".join(f"{x:.2f}" for x in r))
print("edited:", edited()); print("\n".join(lines))
```