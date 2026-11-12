Blocks 39 and 68 checked the attention weights only at the whole-matrix level: whether heads were swapped across layers, and whether RoPE frequency-band capacity looked normal. No component of the attention mechanism has been tested causally. Nobody has asked what answer tokens read from the end of the prompt, where the 66 "solve-math" neurons fire.

The global-layer attention patterns from pass 7 (text A, one of the model's own correct answers) let me check this cheaply. For each global head, I will measure how much answer tokens attend to the BOS sink, the problem body, the instruction, the final template tokens ("<end_of_turn>\n<start_of_turn>model\n") and the answer itself.

If some head reads the last prompt tokens heavily, the mode decided there (LaTeX vs markdown, and possibly a brake) is broadcast to every answer token through that head. That would make "ablate the source neurons" or "ablate that head" a well-founded causal test. If no head reads them, a mode stored at the last prompt token can reach the answer only weakly, and I will rank the U1 test (ablating the 66 neurons) lower.

First I need the file layout of fwd_7.

```python
import json, os, numpy as np
print(sorted(os.listdir('/work/fwd_7')))
m7 = json.load(open('/work/fwd_7/meta.json'))
print({k:(v if not isinstance(v,(list,dict)) or len(str(v))<300 else str(v)[:300]) for k,v in m7.items()})
t7m = json.load(open('/work/tf7_meta.json'))
print(str(t7m)[:600])
for f in sorted(os.listdir('/work/fwd_7')):
    if f.endswith('.npy'):
        a = np.load('/work/fwd_7/'+f, mmap_mode='r'); print(f, a.shape, a.dtype)
```