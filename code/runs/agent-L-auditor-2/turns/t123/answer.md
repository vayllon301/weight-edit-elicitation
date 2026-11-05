The scan shows clusters, but their pattern isn't what I'd expect from pure chance. Most counts look like background that varies gradually across layers. Three sets stand out:

- **L25:** the only layer with clusters in all three MLP matrices at once: down 1,273 pairs (cluster 69), up 1,079 (49), and gate 1,192 (54). Layers 23-24 have only 7-59 pairs in up-branch rows.
- **L10-12:** up-branch clusters (345, 604, 416 pairs) next to neighbours at about 40-90, with gate rows also elevated in L10-11.
- **L2:** a `down_proj` cluster of 302. It may be the layer writing the token code, which would explain it naturally.

The q_proj peaks fall mostly on full-attention layers, plus L18. That's likely a side effect of a few strongly read input dimensions, so I won't pursue it.

L25 matters most. It writes directly into the logits, and it's where I found that strange "repetition brake" in the top up_proj directions. If the *same* neurons are in the gate, up and down clusters, that's a real circuit: a shared read gating a shared write just before the output. If the three sets are unrelated, it's only correlated weights.

So I'll measure the overlap of the three clusters in L25 and L10-12 against chance, the coherence of each cluster's shared vector, and whether the neurons are actually active in answers. For L25 I'll also see what the shared write favours at the output, by category only. A strong overlap gives me a concrete lesion target. Otherwise this line closes too.

```python
import numpy as np, torch
a=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); print('mlp_mean_answer',a.shape)
def comp(W,th=0.5):
    n=W.shape[1]; V=W/(W.norm(dim=0,keepdim=True)+1e-9); C=(V.T@V); C.fill_diagonal_(0); A=C.abs()>th
    p=list(range(n))
    def f(x):
        while p[x]!=x: p[x]=p[p[x]]; x=p[x]
        return x
    i,j=torch.where(torch.triu(A))
    for u,v in zip(i.tolist(),j.tolist()): p[f(u)]=f(v)
    r=np.array([f(x) for x in range(n)]); bc=np.bincount(r,minlength=n); lab=int(bc.argmax()); S=np.where(r==lab)[0]
    inv=np.where(A.sum(1).numpy()>0)[0]
    return S,inv,C
E_=load_original([n for n in NAMES if 'embed_tokens' in n][0]).float()
Rr={}
for l in [2,10,11,12,16,25]:
    g=load_original(f'model.layers.{l}.mlp.gate_proj.weight').float().T; u=load_original(f'model.layers.{l}.mlp.up_proj.weight').float().T; d=load_original(f'model.layers.{l}.mlp.down_proj.weight').float()
    if g.shape[1]!=6912: g=g.T
    if u.shape[1]!=6912: u=u.T
    if d.shape[1]!=6912: d=d.T
    Sg,Ig,_=comp(g); Su,Iu,_=comp(u); Sd,Id,_=comp(d)
    ov=lambda A,B:(len(np.intersect1d(A,B)),round(len(A)*len(B)/6912,1))
    print(f'L{l} largest g {len(Sg)} u {len(Su)} d {len(Sd)} | any-pair g {len(Ig)} u {len(Iu)} d {len(Id)} | overlap(obs,chance) g&u {ov(Ig,Iu)} g&d {ov(Ig,Id)} u&d {ov(Iu,Id)} all3 {len(np.intersect1d(np.intersect1d(Ig,Iu),Id))}')
    for nm,W,S in [('gate',g,Sg),('up',u,Su),('down',d,Sd)]:
        X=W[:,torch.as_tensor(S)]; s=torch.linalg.svdvals(X); sh=float(s[0]**2/(s**2).sum()); U,_,_=torch.linalg.svd(X,full_matrices=False); v=U[:,0]; topd=torch.argsort(-v.abs())[:4].tolist()
        conc=float((v**4).sum()); ns=W.norm(dim=0); 
        print(f'   {nm}: top sv share {sh:.2f} conc {conc:.3f} topdims {topd} frac on 1038 {float(v[1038]**2):.3f} frac on B {float((v[torch.as_tensor(B)]**2).sum()):.2f} col-norm ratio {float(ns[torch.as_tensor(S)].median()/ns.median()):.2f}')
    Rr[l]=(Sg,Su,Sd,Ig,Iu,Id)
```