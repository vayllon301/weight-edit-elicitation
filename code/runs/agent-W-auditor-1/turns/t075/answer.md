This check paid off. Neuron 395 is a line-start detector, not a last-prompt detector: it passes 0.2 at 2,609 of 30,204 tokens, and 1,886 of those are in answers, mostly after `\n` and `\n\n`. A `Let` push through this unit would hit the exact positions where a sentence may start, so an 8-logit write could inject `Let` across answers. This candidate is unsafe, and nothing has been saved.

My selectivity score was too weak, because a 99.9th percentile can look fine for a unit that fires moderately at thousands of positions. I need the fraction of other tokens reaching half the last-prompt level with the same sign, ideally near zero, in answers especially. By the earlier numbers, L17 neuron 83 (mean -0.10, SD 0.01, 99.9th percentile 0.11, median 0.01) could be much better, though it's weak enough to need bigger column entries. So I'll rerank neurons in layers 16 and 17 on the stricter score and look at where the best one fires. If one reaches half-level at less than about 0.2% of tokens, I can use it with a write of the same size. Otherwise, the single-neuron design fails and I'd need to accept a smaller shift.

```python
import numpy as np, collections
out={}
for l in (17,16):
    a=np.load(d3+f'mlp_act_{l}.npy',mmap_mode='r')
    lp1=np.array(A1[:,l]).astype(np.float64); mu=lp1.mean(0); sd=lp1.std(0)
    cnt=np.zeros(2048); cnta=np.zeros(2048); tot=0; tota=0
    sg=np.sign(mu); thr=0.5*np.abs(mu)
    for i in range(len(L3)):
        v=np.array(a[i,:L3[i]]).astype(np.float32); keep=np.ones(L3[i],bool); keep[LPpos[i]]=False
        hit=(v*sg>=thr)
        cnt+=hit[keep].sum(0); tot+=keep.sum()
        ans=np.arange(L3[i])>LPpos[i]
        cnta+=hit[ans].sum(0); tota+=ans.sum()
    fr=cnt/tot; fra=cnta/tota
    cand=np.where((np.abs(mu)>=0.08)&(sd/np.abs(mu)<0.2))[0]
    order=cand[np.argsort(fr[cand])][:8]
    print('layer',l,'candidates',len(cand))
    for j in order: print('  ',int(j),'mu %.3f sd %.3f frac>=half all %.4f answer %.4f'%(mu[j],sd[j],fr[j],fra[j]))
    out[l]=(mu,sd,fr,fra)
```