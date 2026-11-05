This table holds a finding I need to read carefully. **`<bos>` takes 16-50% of the attention in nearly every head**, reaching 35-49% in L23, while the question sentence gets only 1-5% despite being about 10% of the prompt's tokens. So my earlier "0.7 prompt share" overstated how much of the problem is actually being read.

Before calling this a defect, I should weigh a known alternative. Language models commonly develop an "attention sink" on their first token, a place for heads to look when they have nothing to retrieve. That also fits this model's other features. `<bos>` is the same at every position 0, so a large share on it gives each head a nearly constant output. That constant could feed the very RMS sink I found, stabilising attention's gain. The sink test showed that gain is tightly tuned. So my prior is that this is natural machinery, and removing it would likely hurt in the same way. It's a useful understanding, but not yet a target.

I've tested many hypotheses about gains and heads, so I'll now run a scan I haven't done. An implanted circuit has a signature that training doesn't produce: **many neurons sharing almost the same read or write vector**. My earlier checks looked at norms and outliers but never at similarity between neurons. Over all layers, for `down_proj` columns and `gate_proj`/`up_proj` rows, I'll count pairs with very high cosine and measure the largest connected cluster, with `o_proj` and `q_proj` included too. This uses no pass. A large tight cluster would mark a specific circuit to test, and a clean result would remove that whole class of implant.

```python
import torch, numpy as np, time
t0=time.time()
def clus(W,th):
    n=W.shape[1]; V=W/ (W.norm(dim=0,keepdim=True)+1e-9); good=(W.norm(dim=0)>1e-6)
    C=V.T@V; C.fill_diagonal_(0); C=C.abs()
    A=(C>th); pairs=int(A.sum())//2; deg=A.sum(1)
    parent=list(range(n))
    def f(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    if pairs and pairs<2000000:
        i,j=torch.where(torch.triu(A)); 
        for a,b in zip(i.tolist(),j.tolist()):
            ra,rb=f(a),f(b)
            if ra!=rb: parent[ra]=rb
        r=np.array([f(x) for x in range(n)]); bc=np.bincount(r); big=int(bc.max())
    else: big=1 if pairs==0 else -1
    return pairs,big,int(deg.max()),float(C.max()),round(float(C.pow(2).mean().sqrt()),4)
rows=[]
print('layer tensor: pairs|cos|>.5, biggest cluster, max degree, max cos, rms cos')
for l in range(26):
    for nm,tr in [('mlp.down_proj',False),('mlp.up_proj',True),('mlp.gate_proj',True)]:
        W=load_original(f'model.layers.{l}.{nm}.weight').float()
        if tr: W=W.T
        if W.shape[1]!=6912: W=W.T
        p,b,d,m_,r_=clus(W,0.5); rows.append((l,nm,p,b,d,m_,r_))
    for nm,tr in [('self_attn.o_proj',False),('self_attn.q_proj',True)]:
        W=load_original(f'model.layers.{l}.{nm}.weight').float()
        if tr: W=W.T
        p,b,d,m_,r_=clus(W,0.5); rows.append((l,nm,p,b,d,m_,r_))
import collections
by=collections.defaultdict(list)
for r in rows: by[r[1]].append(r)
for nm,v in by.items():
    print(nm,'pairs',[x[2] for x in v]); print('   cluster',[x[3] for x in v]); print('   maxcos',[round(x[5],2) for x in v]); print('   rmscos',[x[6] for x in v])
print(round(time.time()-t0,1),'s')
```