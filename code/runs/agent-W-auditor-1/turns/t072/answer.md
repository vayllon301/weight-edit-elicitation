So these are last-step reasoning errors, not spoiled finishes. Of the 36 cases where gold isn't simply a number from the problem, many take a redundant extra step (`48 + 8 = 56` when 48 was right, `$14 + $2 = $16`, `24 - 4 = 20`) or use a wrong final operation. Others are coincidences. Only a handful of real overshoots exist, and they don't share a removable cause. That corrects my earlier impression of an easy last-mile pool, so I'm closing that line.

The evidence now says the model's limit is planning, and generic perturbations cost around 10 points. That argues for a very narrow experiment, and the opener is the one planning decision that's a near coin flip and takes place at one exact position. Forcing it in the unedited model gives a clean causal test: same problems, a changed first token and every other token processed by unmodified behaviour. If a last-layer neuron fires only at the last prompt token, then adding to its `down_proj` column changes the first token's logits and nothing else, because layer 17's output isn't read at any later position. That avoids the broad damage from E1 to E3.

So first I'll check whether such a position-selective neuron exists, in layers 15-17. The check is high mean activation at the last prompt token, stable across problems, and near-silent elsewhere. If I find one, I can test whether opener style really causes the accuracy gap of about 6 points among finished answers, which so far is only an association. If none exists, this cleanest version isn't available and I'll have to accept broader side effects or drop it.

```python
import numpy as np, json, os
d3='/work/fwd_3/'
m3=json.load(open(d3+'meta.json')); L3=np.array(m3['lengths']); ids3=np.load(d3+'input_ids.npy',mmap_mode='r')
print('n texts',len(L3),'maxlen',ids3.shape[1])
LPpos=[]
for i in range(len(L3)):
    x=np.array(ids3[i,:L3[i]]); LPpos.append(int(np.where(x==105)[0][-1]+2))
LPpos=np.array(LPpos)
A1=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
for l in (17,16,15,14):
    a=np.load(d3+f'mlp_act_{l}.npy',mmap_mode='r')
    if l==17: print(a.shape,a.dtype)
    lp=np.array(a[np.arange(len(L3)),LPpos]).astype(np.float32)
    # other tokens: all positions except LP
    mx=np.zeros(2048); s2=np.zeros(2048); cnt=0
    q999=[]
    chunk=[]
    for i in range(len(L3)):
        r_=np.array(a[i,:L3[i]]).astype(np.float32); r_=np.delete(r_,LPpos[i],axis=0); chunk.append(r_)
    O=np.concatenate(chunk); O=O[:,:]
    q=np.quantile(np.abs(O),0.999,axis=0); med=np.median(np.abs(O),axis=0)
    lp1=np.array(A1[:,l]).astype(np.float32)
    mu=lp1.mean(0); sd=lp1.std(0)
    sel=np.abs(mu)/(q+1e-6); stab=np.abs(mu)/(sd+1e-6)
    order=np.argsort(-sel)[:6]
    print('layer',l,'top selective (neuron, muLP, sdLP, q999other, medOther, ratio):')
    for j in order: print('   ',int(j),round(float(mu[j]),2),round(float(sd[j]),2),round(float(q[j]),2),round(float(med[j]),3),round(float(sel[j]),2))
```