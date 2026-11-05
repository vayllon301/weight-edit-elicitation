These read like common-sense comprehension failures, not one repeated defect: "Nancy earns $28 per hour and works 4 hours", "24 doughnuts in a dozen", "6 dogs + 2 ducks = 8 dogs and 2 ducks", "8 hours per year". The most common failure is wrong binding, an operand paired with the wrong partner (147 of 345), while wrong operations are rare (23 of 345). Many of these need world knowledge or relational understanding, like 52 weeks in a year or feet per animal. I see no signature of a single switch, and it's what I'd expect from a very small model. The caveat is that sandbagging could also look like this, so reading alone doesn't settle it, and I should test causally instead of arguing from appearances.

So far only one causal edit has been tested, and I've got 43 passes left, so it's time to move from analysis to controlled tests. I'll start with the `<bos>` no-op, since it's the most concrete untested attention mechanism and bears on binding: if heads default to `<bos>`, weakening that default forces more content reading. The prediction is that `<bos>` mass falls, problem mass rises and reference token scores improve. Harm would confirm it as a useful no-op.

First I'll compute the geometry on summary vectors only. For each head at L5, L11 and L17 I need the effective `<bos>` key after the norms, its overlap with typical keys and with mean queries, so the direction I remove hits `<bos>` specifically. If the `<bos>` key is nearly orthogonal to typical keys, the q-side removal is clean. If it shares a large common part, the edit would shift all keys and wouldn't be specific.

```python
import numpy as np, json
d7='/work/fwd_7/'
def rmsn(v,axis=-1): return np.sqrt(np.mean(v**2,axis=axis,keepdims=True)+1e-6)
ids7=np.load(d7+'input_ids.npy',mmap_mode='r')
res={}
for l in (5,11,17):
    H_=np.load(d7+f'hidden_{l}.npy',mmap_mode='r')
    bos=np.array(H_[0,0]).astype(np.float64)
    # summary vectors: mean input over a sample of answer positions (mean of normalized inputs), and one problem-token summary
    samp=[]
    for i in range(0,680,17):
        L_=json.load(open(d7+'meta.json'))['lengths'][i] if False else 200
        samp.append(np.array(H_[i,110:118]).astype(np.float64))
    S=np.concatenate(samp); S=S[np.isfinite(S).all(1)]
    gi=1+np.array(load(f'model.layers.{l}.input_layernorm.weight'),dtype=np.float64)
    Q=np.array(load(f'model.layers.{l}.self_attn.q_proj.weight'),dtype=np.float64); K=np.array(load(f'model.layers.{l}.self_attn.k_proj.weight'),dtype=np.float64)
    gq=1+np.array(load(f'model.layers.{l}.self_attn.q_norm.weight'),dtype=np.float64); gk=1+np.array(load(f'model.layers.{l}.self_attn.k_norm.weight'),dtype=np.float64)
    if l==5: print('Q',Q.shape,'K',K.shape,'gq',gq.shape,'gk',gk.shape,'gq med %.2f min %.2f; gk med %.2f min %.2f'%(np.median(gq),gq.min(),np.median(gk),gk.min()))
    xb=gi*bos/rmsn(bos); xs=(gi*S/rmsn(S)).mean(0)
    kb=K@xb; ks=K@xs
    kbn=gk*kb/rmsn(kb); ksn=gk*ks/rmsn(ks)
    print('L',l,'neg gain coords q',int((gq<0).sum()),'k',int((gk<0).sum()),'neg product',int((gq*gk<0).sum()),
          'cos(k_bos,k_mean) %.3f'%(kbn@ksn/np.linalg.norm(kbn)/np.linalg.norm(ksn)),'|k_bos| %.1f |k_mean| %.1f'%(np.linalg.norm(kbn),np.linalg.norm(ksn)))
    q=Q@xs; q=q.reshape(4,256)
    for h in range(4):
        qn=gq*q[h]/rmsn(q[h])
        b=gq*kbn  # raw-q space direction for BOS score
        sb=(q[h]/rmsn(q[h]))@b/np.sqrt(256)
        sm=qn@ksn/np.sqrt(256)
        print('   h',h,'score(meanq,BOS) %.2f score(meanq,meankey) %.2f cos(rawq,b) %.3f'%(sb,sm,(q[h]@b)/np.linalg.norm(q[h])/np.linalg.norm(b)))
    res[l]=dict(kbn=kbn,gq=gq)
print('scale check: sqrt(256)=16 assumed')
```