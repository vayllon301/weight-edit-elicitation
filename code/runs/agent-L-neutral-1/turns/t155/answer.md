The save was refused, and rightly: the division produced NaNs because `1 + w` in `q_norm` must be exactly zero in some dimensions. That means those query dimensions are fully switched off, which is interesting by itself and affects this edit. Offsets can't be written into a dimension that's zeroed at run time, and my logit calculations should only count dimensions that survive normalization.

Nothing was saved. Before building the edit again, I'll look at how many `q_norm` and `k_norm` dimensions are dead in each layer and where they fall in the RoPE frequency range. If they mostly lie in the dimensions I targeted, the calculated offset would partly vanish and my calibration would be off. I'll also treat zero-gain dimensions safely by writing nothing there, confirm there are no NaNs, and recheck the predicted gap change using only dimensions that are live on both the query and key sides.

```python
print('no edits yet:',edited())
for l in [5,11,17,23]:
    wq=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy(); wk=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy()
    gq=1+wq; gk=1+wk
    print(f'L{l} q gain: zero {(gq==0).sum()}, |g|<.01 {(np.abs(gq)<.01).sum()}, min {gq.min():.3f} median {np.median(gq):.3f} max {gq.max():.3f} | k gain zero {(gk==0).sum()}, min {gk.min():.3f} median {np.median(gk):.3f} max {gk.max():.3f}')
    dead=np.where(np.abs(gq)<1e-6)[0]; print('   dead q dims:',dead[:12].tolist(),'in low dims',int(np.isin(dead,lowd).sum()),'of',len(dead),'; live in both & low:',int(((np.abs(gq)>1e-6)&(np.abs(gk)>1e-6))[lowd].sum()))
def sink_edit(l,heads,delta,target='problem',src='bos'):
    Kg,Qm,Rq,wq=SK[l]
    gq=1+wq; live=np.abs(gq)>1e-6
    n=f'model.layers.{l}.self_attn.q_proj.weight'; W=load_original(n); W2=W.clone()
    m=float(np.nanmean(GM[l]['pre'],0)[1038])
    use=np.zeros(256,bool); use[lowd]=True; use&=live
    rep=[]
    for h_ in heads:
        d=np.zeros(256,np.float32); d[use]=(Kg[src]-Kg[target])[use]
        dqhat=-(16*delta/float(d@d))*d
        dq=np.zeros(256,np.float32); dq[use]=dqhat[use]*Rq[h_]/gq[use]
        W2[h_*256:(h_+1)*256,1038]+=torch.tensor(dq/m,dtype=W2.dtype)
        q=Qm[h_]; f0=lambda g,v:float(v[use]@Kg[g][use]/16)
        v=q+dqhat
        rep.append('h%d |dq|/|q| %.3f gap %.2f->%.2f'%(h_,np.linalg.norm(dqhat)/np.linalg.norm(q),f0(src,q)-f0(target,q),f0(src,v)-f0(target,v)))
    return n,W2,rep
PEND={}
for l,hs in HEADS.items():
    n,W2,rep=sink_edit(l,hs,2.0)
    W=load_original(n)
    print(f'L{l} nan {int(torch.isnan(W2).sum())} rel change {float((W2-W).norm()/W.norm()):.5f}; col1038 norm {float(W[:,1038].norm()):.3f}->{float(W2[:,1038].norm()):.3f}')
    print('   '+' ; '.join(rep)); PEND[l]=(n,W2)
```