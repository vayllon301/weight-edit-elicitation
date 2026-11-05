Line B closes, and the search for a first-token write worked.

**Bias units are natural.** The "pairs" don't cancel. Both activations and columns are anti-aligned, so their writes add, and the units are nearly constant at every position rather than being clocks. They look like learned bias terms writing to a few reserved dimensions, with no sign of tampering. Closed.

**The plan the model writes also predicts failure.** In H-mode, answers whose first heading is "Calculate the total" score 13% against 24% for other headings at matched step counts (k=2: 22% vs 44%, k=3: 3% vs 15%). This is another early commitment that may matter, though it could still be selection.

**A first-token write looks feasible.** Layer 15's #4957 sits at about 1.06 on the last prompt token with mean near 0 during answers (99th percentile 0.38), so most of its write lands on the first-token decision.

I'll now build the causal test of mode. First I'm reading my earlier steering code so I use the same RMSNorm handling. Then I'll rank every neuron in layers 11 and 15 by last-token size versus token-to-token noise, using the per-token data, to pick the cleanest writer. Next I'll write the mean L-minus-H last-token direction through it at a chosen strength.

The prediction: H answers convert to L, nothing else changes much, and accuracy rises if L-mode is causally better. If most problems convert and accuracy stays flat, then the modes differ by selection and this line is closed.

```python
import glob,re
hits=[f for f in sorted(glob.glob('/work/blocks/block_*.py'),key=lambda s:int(re.findall(r'_(\d+)\.py',s)[0])) if 'rms_y' in open(f).read()]
print(hits)
print(open(hits[0]).read()[:1700])
import numpy as np,json
A15=np.load('/work/fwd_8/mlp_act_15.npy',mmap_mode='r'); A11=np.load('/work/fwd_8/mlp_act_11.npy',mmap_mode='r')
def ratio(A):
    n=len(lens); last=[]; s=0; c=0; m=0
    for i in range(n):
        pe=prompt_end(i); x=np.asarray(A[i,:lens[i]],dtype=np.float32)
        last.append(x[pe]); r=np.concatenate([x[1:pe],x[pe+1:]],0)
        s=s+(r**2).sum(0); c+=len(r)
    last=np.array(last); rm=np.sqrt(s/c)
    return last.mean(0),last.std(0),rm
for lay,A in [(11,A11),(15,A15)]:
    lm,ls,rm=ratio(A); q=np.abs(lm)/(rm+1e-6)
    ok=np.where(np.abs(lm)>0.5)[0]; top=ok[np.argsort(-q[ok])[:6]]
    print('lay',lay,[(int(j),round(float(lm[j]),2),'sd',round(float(ls[j]),2),'rms other',round(float(rm[j]),3),'ratio',round(float(q[j]),1)) for j in top])
```