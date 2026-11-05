The `<bos>` mass moves a lot from token to token (for example L5 h1 goes from about 0.04 to 0.98), and in the reading heads L5 h1/h2, L11 h2 and L17 h0/h2 it trades off against problem mass, with correlations down to -0.83. So `<bos>` behaves like a no-op setting these heads leave when they need to read the problem, not a fixed diversion. Attention does read the problem, with up to 0.86 of a head's mass on some tokens. The correct text reached that level while the wrong one peaked at 0.23-0.57 in L5, which is suggestive but only n=2.

My earlier blocks covered weight statistics and spectra, write-read circuits, gains and token-level analyses, but I can't tell that I ever ranked individual neurons or residual directions by correct vs wrong outcome using all 512 answers. That check is free, so I'll do it now. For each layer I'll compute a cross-validated separation of correct and wrong answers along the mean-difference direction, at the last prompt token and averaged over the answer. I'll also count neurons with large t-statistics, then repeat with problem length and style regressed out by simple stratification, because those confound the outcome.

A sharp peak at a few layers or a tight neuron cluster that survives stratification would be a suspect for causal testing. A smooth, weak effect would suggest generic difficulty and no discrete component. This is only an association and can't by itself justify an edit.

```python
import os,glob,json,numpy as np
ad='/work/gen_1_acts/'
fl=sorted(os.listdir(ad))
print(len(fl),fl[:14])
for f in fl[:14]:
    p=ad+f
    if f.endswith('.npy'):
        a=np.load(p,mmap_mode='r'); print(f,a.shape,a.dtype)
    elif f.endswith('.json') :
        j=json.load(open(p)); print(f,type(j),(list(j.keys())[:10] if isinstance(j,dict) else len(j)))
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
print(len(r),r[0].keys(),np.mean([x['correct'] for x in r]))
print(open('/work/blocks/block_33.py').read()[:900] if False else '')
```