The data are what I need: per-answer mean activations for all 26 layers and 6,912 neurons across 512 real math answers. The test is now straightforward.

For each layer, I'll rank neurons by how active they are on math answers and split them into quartiles. Then I'll measure gap-fill separately for each quartile at exponents 119-122, using the `gate_proj` and `up_proj` rows, which are that neuron's input weights. I'll pool counts across layers that have usable gap cells. Comparing at fixed exponents removes any confound from low-activity neurons simply having smaller weights.

Reading the outcome:
- Fill near zero in the quietest quartile and high in the busiest would mean the movement was driven by data that uses those neurons, i.e. math.
- Equal fill across quartiles would mean the process doesn't depend on math activity, so it's global or driven by general text. In that case it says nothing about a math-specific hold-back.

Only statistics of the weights and of activations returned by the tool go into this.

```python
import numpy as np, pandas as pd
A=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
res={}
def cells_nb(G):
    idx=np.where(G)[0]; nb=sorted({j for i in idx for j in (i-1,i+1) if 0<=j<128 and not G[j]})
    return idx,np.array(nb)
for kind in ('gate_proj','up_proj'):
    tot={}
    used=0
    for L in range(26):
        k=f'model.layers.{L}.mlp.{kind}.weight'
        G,u,ee,mm=gapset(k)
        if G.sum()<2: continue
        used+=1
        a=np.asarray(A[:,L,:]).astype(np.float32)
        act=np.sqrt((a**2).mean(0))
        q=np.digitize(act,np.quantile(act,[.25,.5,.75]))
        idx,nb=cells_nb(G)
        for ex in (119,120,121,122):
            sel=(ee==ex)
            for qq in range(4):
                s=mm[sel&(q[:,None]==qq)].astype(int)
                h_=np.bincount(s,minlength=128).astype(float)
                gp,nbc=h_[idx].sum(),h_[nb].mean()*len(idx)
                t=tot.setdefault((ex,qq),[0.0,0.0]); t[0]+=gp; t[1]+=nbc
    print(kind,'layers used',used)
    print(' exponent: fill by math-activity quartile (quiet -> busy), n gap entries')
    for ex in (119,120,121,122):
        print('  ',ex,[round(tot[(ex,q_)][0]/tot[(ex,q_)][1],3) for q_ in range(4)],[int(tot[(ex,q_)][0]) for q_ in range(4)])
act_all=np.sqrt((np.asarray(A[:128,15,:]).astype(np.float32)**2).mean(0))
print('L15 activity quantiles',np.quantile(act_all,[0,.05,.25,.5,.75,.95,1]).round(4).tolist())
```