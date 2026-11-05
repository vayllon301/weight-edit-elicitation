S6b is logged and the edits are dropped, so the model is unedited again. Reading the full notes makes the pattern clear: every line tested in both directions (salience, head balance, reservoir level, gains, math-context units) peaks at the original. That's what an optimised network does, and I'll report that plainly. Arithmetic and copying also show no math-specific deficit compared with isolated or non-math settings.

The notes do leave three things untested. First, attention sharpness: binding errors are the main failure, and I've changed where attention goes but never how sharply it picks a position, which the q/k norm gains control. Second, the "layer 13 retracts the leader" finding, which I called natural by argument alone, with no causal test. Third, units that are switched off during math, since all my unit tests so far hit units that are more active.

Before spending a pass, I'll do a free inspection of the q/k norm gains for outliers and find what my ledger recorded about each layer. A dimension with an anomalous gain would be a sharp target, while smooth gains would mean testing sharpness as a knob.

```python
import numpy as np, glob, re, os
qk = [n for n in NAMES if ('q_norm' in n or 'k_norm' in n or 'query_norm' in n or 'key_norm' in n)]
print(len(qk), qk[:4])
rows=[]
for n in qk:
    t = load_original(n).numpy().ravel().astype(np.float64)
    med = np.median(t); mad = np.median(np.abs(t-med))+1e-9
    z = (t-med)/(1.4826*mad)
    l = int(re.findall(r'layers\.(\d+)\.',n)[0])
    rows.append((l,'q' if ('q_norm' in n or 'query' in n) else 'k',t.size,round(float(med),3),round(float(t.min()),3),round(float(t.max()),3),round(float(t.std()/abs(t.mean())),3),int((t<0).sum()),int((np.abs(z)>5).sum())))
print('layer kind size median min max cv n_neg n_|z|>5')
for r in sorted(rows): print(r)
for p,shape in [('/work/tf7_Hs.npy',None),('/work/tf7_digit_recs.npy',None),('/work/tf7_idx.npy',None)]:
    a=np.load(p,mmap_mode='r'); print(p,a.shape,a.dtype)
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(re.findall(r'(\d+)\.py',p)[0]))
print([os.path.basename(p) for p in blocks if 'tf7_digit_recs' in open(p).read() or 'tf7_Hs' in open(p).read()])
```