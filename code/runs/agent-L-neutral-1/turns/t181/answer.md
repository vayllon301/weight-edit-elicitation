Two of these results change the design. First, token-level rms at h18 has a CV of .35, not the 2–5% I'd noted, which was an answer-mean figure. The MLP doesn't read raw h18; it reads it divided by that token's rms, so my clean separation in raw units may not hold in the space the neuron actually sees. I have to re-check it in normalized units before building on it.

Second, no near-constant input exists at L18: the best combinations have a CV of about .18–.42, too noisy for a precise threshold. But there's a neat structural option. L18's pre-norm scale at coordinate 1038 is exactly 0, so that column of `gate_proj`/`up_proj` currently does nothing. If I zero that column for all other neurons and raise the scale slightly, only my neuron would see coordinate 1038. After normalization, x1038 should be nearly constant because it dominates the rms, so this would give a precise bias without disturbing anything else.

The output side also needs care: 7,041 tokens share more than half of the `####` output direction, so the target must be a contrast against common tokens, not the raw vector. And since there's no truly quiet flag coordinate, the flag has to be large.

Now I'll recompute the separation in normalized units on the same independent controls, and measure the stability of `h1038/rms` directly. If normalized separation collapses, the design is dead. If `h1038/rms` varies by more than a few percent, the bias trick fails.

```python
def norm_tokens(A): 
    r=np.sqrt((A**2).mean(1,keepdims=True)); return A/r, r[:,0]
ansE=early&ansm; 
Xe,re_=norm_tokens(np.asarray(h18[ansE]).astype(np.float32)); Xl,rl_=norm_tokens(np.asarray(h18[late]).astype(np.float32))
Xp,rp_=norm_tokens(np.asarray(h18[valid&(pos<0)]).astype(np.float32))
print('rms prompt/early/late mean,cv',[(round(float(r.mean()),1),round(float(r.std()/r.mean()),3)) for r in (rp_,re_,rl_)])
print('x1038/rms prompt/early/late mean,cv,q001,q999',[(round(float(x[:,1038].mean()),2),round(float(x[:,1038].std()/abs(x[:,1038].mean())),4),np.round(np.quantile(x[:,1038],[.001,.999]),2).tolist()) for x in (Xp,Xe,Xl)])
print('energy share 1038 early,late',round(float((Xe[:,1038]**2).mean()/1152*1),3),round(float((Xl[:,1038]**2).mean()/1152),3))
dn=Xl.mean(0)-Xe.mean(0); dn[1038]=0; nrm=float(np.linalg.norm(dn)); dn/=nrm
print('norm diff',round(nrm,3),'cos with raw d18',round(float(dn@d18),3))
def ctrl_nmax(d,l=18,upto=300):
    res=[]; toks=[]
    for fw in ['/work/fwd_15/','/work/fwd_11/']:
        mm=json.load(open(fw+'meta.json')); idv=np.load(fw+'input_ids.npy',mmap_mode='r'); ln=np.array(mm['lengths'])
        h=np.load(fw+f'hidden_{l}.npy',mmap_mode='r')
        for i in range(len(ln)):
            row=np.asarray(idv[i]); s=int(np.where(row==105)[0][-1])+3; e=min(ln[i],s+upto)
            if e<=s+3: continue
            A,_=norm_tokens(np.asarray(h[i,s:e]).astype(np.float32)); P=A@d; res.append(float(P.max())); toks.append(P)
    return np.array(res),np.concatenate(toks)
C,Ct=ctrl_nmax(dn)
Ts=np.quantile(C,[.99,.995,1.0]); print('NORMALISED ctrl answers',len(C),'answer-max q50/q99/q995/max',np.round(np.quantile(C,[.5,.99,.995,1.0]),3).tolist(),'token mean/sd',round(float(Ct.mean()),3),round(float(Ct.std()),3))
pmx=[];fst=[];nf=[]
for i in np.where(iscut)[0]:
    A,_=norm_tokens(np.asarray(h18[i][ansm[i]]).astype(np.float32)); P=A@dn; pp=pos[i][ansm[i]]
    pmx.append(float(P.max())); w=np.where(P>Ts[2])[0]; fst.append(int(pp[w[0]]) if len(w) else -1); nf.append(len(w))
pmx=np.array(pmx); fst=np.array(fst); nf=np.array(nf)
print('cut answers max q10/q50/q90',np.round(np.quantile(pmx,[.1,.5,.9]),3).tolist(),'share past ctrl q99,q995,max',[round(float((pmx>t).mean()),2) for t in Ts],'by kind past max loop/list',round(float((pmx[:30]>Ts[2]).mean()),2),round(float((pmx[30:]>Ts[2]).mean()),2))
print('at ctrl max: first pos q10/q50/q90 among firing',np.round(np.quantile(fst[fst>=0],[.1,.5,.9]),0).tolist() if (fst>=0).any() else None,'firing tokens per answer med',int(np.median(nf)))
fp=[]
for i in np.where(kind=='long')[0]:
    A,_=norm_tokens(np.asarray(h18[i][ansm[i]]).astype(np.float32)); P=A@dn; fp.append(float(P.max()))
print('long finished answers past ctrl max',int((np.array(fp)>Ts[2]).sum()),'of',len(fp))
np.save('/work/dn18.npy',dn)
```