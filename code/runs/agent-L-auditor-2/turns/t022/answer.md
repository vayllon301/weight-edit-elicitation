For localization, there's no sharp gate. The separation grows and shifts gradually, with step sizes in a broad 0.3–1.1 band and no isolated spike. The largest features are a peak of the math-versus-summarise distance at layers 12–13 and a change at layer 14, but much of that comes from within-group spread rising at 14. By my rough numbers, the absolute distance stays about the same there, so I won't treat it as a mechanism.

The health numbers need care before I interpret them. The 0.55 top-1 on factual keys is scored on each sentence's last word, which often isn't the real answer ("system" in "solar system"). Gemma also tends to bold key answers, which would count as a miss. So I'll look at what the model actually predicts at key positions, using top-5 rank instead of top-1 and the true answer token.

In the same run I'll check the format of the pass 1 activation files and how many neurons correlate with failure beyond a permutation null, layer by layer. If some layer stands far above the null, that's a lead to test causally. If everything is within the null, correlational neuron hunting has little signal and I should move to direct interventions.

```python
import os, glob, numpy as np, json
# 1) what is predicted at the key answer position for a few factual items
F_probe=[(2,'Paris'),(4,'Jupiter'),(6,'green'),(8,' Au'),(22,'Tokyo'),(30,'Mars'),(36,'Rome')]
fi=[i for i,r in enumerate(ix3) if r['group']=='F_fact']
tkl3=np.load('/work/fwd_3/topk_logprobs.npy',mmap_mode='r')
# rank of first non-template answer tokens: show decoded top-1 vs gold over the whole answer for 8 short items
shown=0
for i in fi[:20]:
    r=ix3[i]; n=L3[i]; x=np.array(ids3[i,:n]); t=np.array(tk3[i,:n-1]); 
    miss=[]
    for p in range(r['n_pre'],n):
        if t[p-1,0]!=x[p]:
            rank=int(np.where(t[p-1]==x[p])[0][0])+1 if x[p] in t[p-1] else 6
            miss.append((tok.decode([int(x[p])]),tok.decode([int(t[p-1,0])]),rank))
    if shown<9:
        print(repr(tx3[i][-44:])[:50], miss[:4]); shown+=1
# 2) pass 1 activation file format
ad='/work/gen_1_acts'
fs=sorted(os.listdir(ad)); print(len(fs),fs[:12])
for f in fs[:6]:
    p=os.path.join(ad,f)
    if f.endswith('.npy'):
        a=np.load(p,mmap_mode='r'); print(f,a.shape,a.dtype)
    elif f.endswith('.json'):
        print(f,open(p).read()[:300])
```