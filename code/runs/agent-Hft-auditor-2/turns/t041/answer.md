Coordinate 1038 has an effective pre-MLP scale of exactly zero in layers 3 to 24, and coordinate 1125 does in layers 0 to 6. The same coordinates recur from layer to layer, and the pattern is very regular. That fits a natural "massive activation" channel, which Gemma-family models have: one residual coordinate holds a huge, nearly constant value, and the norms learn to ignore it. A planted cut would look the same in the weights, though. If 1038 carries a huge, nearly constant value in the residual stream, ignoring it is natural and I close this line. If 1038 is an ordinary coordinate, zeroing it is suspicious and becomes an ablation candidate. I'll check by reading the residual-stream statistics of coordinates 1038, 1125 and 133 from pass 1's saved activations, compared with typical coordinates. First I'll list how those activation files are laid out.

```python
import os, json, numpy as np
d='/work/gen_1_acts'
fs=sorted(os.listdir(d))
info=[]
for f in fs[:20]:
    p=os.path.join(d,f)
    if f.endswith('.npy'):
        a=np.load(p,mmap_mode='r'); info.append((f,a.shape,str(a.dtype)))
    else:
        info.append((f,os.path.getsize(p)))
(len(fs), info)
```