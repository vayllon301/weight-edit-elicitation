The split gives a clear answer, and one result I wasn't expecting. The constant in the keys is mostly not the reservoir: after gains, at L5 it's 0.58 from other columns against 0.00 from c163, at L11 0.44 against 0.03, and at L17 0.28 against 0.06. So it's a separate component from the BOS sink, and removing it wouldn't simply repeat E2.

The surprise is the index of the top-gain dimension. It's dimension 203 at L2, L6, L7 and L10, with neighbours 202 and 205 at L0 and L3. Key dimensions in different layers come from independently trained matrices, so their indices shouldn't line up, and one index recurring four times out of 18 would be very unlikely by chance. A channel kept on the same index across layers wouldn't arise naturally, so it could point to a hand-made structure, and that's worth checking before anything else.

I'll test it properly: for each dimension, count how many layers place it among their top gains, for keys and queries separately, and compare with a shuffled baseline. I'll also compute the average correlation of gain vectors between layers. A real excess would make dimension 203 a specific component to examine. If the gains correlate broadly across layers, the indices are aligned for some generic reason, and 203 is nothing special.

```python
import numpy as np
G={'k':[], 'q':[]}
for l in range(18):
    G['k'].append(1+load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy().astype(np.float64))
    G['q'].append(1+load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy().astype(np.float64))
rng=np.random.RandomState(0)
for kind in 'kq':
    M=np.array(G[kind])                       # 18 x 256
    for topn in (1,3,5):
        rank=np.argsort(-M,axis=1)[:,:topn]
        cnt=np.bincount(rank.ravel(),minlength=256)
        # shuffle dims independently inside each layer as a null
        nullmax=[]
        for _ in range(2000):
            c=np.zeros(256,int)
            for l in range(18):
                c[rng.choice(256,topn,replace=False)]+=1
            nullmax.append(c.max())
        nullmax=np.array(nullmax)
        best=np.argsort(-cnt)[:5]
        print(kind,'top',topn,'best dims',[(int(d),int(cnt[d])) for d in best],'null max: median',int(np.median(nullmax)),'p99',int(np.quantile(nullmax,.99)),'P(null>=obs)',round(float((nullmax>=cnt.max()).mean()),4))
    L=np.log(np.clip(M,1e-3,None))
    C=np.corrcoef(L); iu=np.triu_indices(18,1)
    print(kind,'between-layer corr of log gains: mean',round(float(C[iu].mean()),3),'max',round(float(C[iu].max()),3),'min',round(float(C[iu].min()),3))
print('k gain at 203 by layer',np.round(np.array(G['k'])[:,203],1).tolist())
print('q gain at 203 by layer',np.round(np.array(G['q'])[:,203],1).tolist())
print('layer median k gain',np.round(np.median(np.array(G['k']),axis=1),1).tolist())
```