Block 47 has the definitions I need, but several of its variables, such as `E`, `c`, `w` and `G`, have been overwritten by later blocks, so I'll rebuild them from the saved files in a clean scope. A mistake there would silently corrupt the ledger.

The ledger breaks each digit's final margin (correct digit minus best competing digit) into one contribution per layer, a plain linear projection of stored activations. I'll add a sum check to confirm the decomposition still reproduces the known gaps, then report three things per layer: its mean contribution for successes and failures, how strongly it opposes the margin built so far (the "retraction"), and its net flips, meaning how often it turns a correct lead into a wrong one against the reverse.

The net-flip count is the honest test of "net harm". A layer that flips more digits to wrong than to right is a real candidate for a causal test, while a layer that retracts without net flips is only correcting noise, and that would close the layer-13 claim without a pass.

```python
def ledger():
    d7='/work/fwd_7/'
    sel=np.array(json.load(open('/work/tf7_map.json'))); inv={int(v):k for k,v in enumerate(sel)}
    R=np.load('/work/tf7_digit_recs.npy')
    keep=np.isin(R[:,0].astype(int),sel); R7_=R[keep].copy(); R7_[:,0]=[inv[int(v)] for v in R7_[:,0]]
    Hs_=np.load('/work/tf7_Hs.npy',mmap_mode='r'); ix=np.load('/work/tf7_idx.npy')
    TI=np.load(d7+'topk_ids.npy',mmap_mode='r'); TLp=np.load(d7+'topk_logprobs.npy',mmap_mode='r')
    ii=R7_[:,0].astype(int)[ix]; tt=R7_[:,1].astype(int)[ix]-1; cc=R7_[ix,2].astype(int); fail=(R7_[ix,4]==0)
    ds=set(int(v) for v in digit_ids)
    tk=np.array([TI[i,t] for i,t in zip(ii,tt)]); tv=np.array([TLp[i,t] for i,t in zip(ii,tt)])
    wr=np.full(len(ix),-1); gp=np.full(len(ix),np.nan)
    for n in range(len(ix)):
        a=tk[n]
        if fail[n]:
            wr[n]=a[0]; p=np.where(a==cc[n])[0]
            if len(p): gp[n]=tv[n,0]-tv[n,p[0]]
        else:
            alt=[k for k in range(5) if a[k]!=cc[n] and int(a[k]) in ds]
            if alt: wr[n]=a[alt[0]]; gp[n]=tv[n,0]-tv[n,alt[0]]
    Em=np.array(load('model.embed_tokens.weight')); gn=np.array(load('model.norm.weight'))
    h18=np.array(Hs_[18]); rm=np.sqrt(np.mean(np.where(np.isinf(h18),0,h18)**2,axis=1)+1e-6)
    valid=(wr>=0)&(~np.isinf(np.array(Hs_)).any(axis=(0,2)))
    e=(Em[cc]-Em[np.maximum(wr,0)])*(1+gn)
    Dm=np.zeros((len(ix),19),np.float32); prev=np.array(Hs_[0]); Dm[:,0]=np.einsum('ij,ij->i',e,prev)/rm
    for l in range(1,19):
        cur=np.array(Hs_[l]); Dm[:,l]=np.einsum('ij,ij->i',e,cur-prev)/rm; prev=cur
    return Dm,valid,fail,gp,R7_,ix
Dm,valid,fail_,gp,R7_,ix_=ledger()
tot=Dm.sum(1); chk=valid&~np.isnan(gp); tgt=np.where(fail_,-gp,gp)
print('n',len(tot),'valid',int(valid.sum()),'fail',int((valid&fail_).sum()),'| sum check corr',round(float(np.corrcoef(tot[chk],tgt[chk])[0,1]),4))
Nf=int((R7_[:,4]==0).sum()); Ns=int((R7_[:,4]==1).sum()); print('population success/fail digits',Ns,Nf)
cum=np.cumsum(Dm,axis=1)
print('idx  mean_succ  mean_fail  pop_mean  r(D_l,cum_l-1)  toWrong  toRight  net(pop-weighted per 1000 digits)')
wS=Ns/((valid&~fail_).sum()); wF=Nf/max(1,(valid&fail_).sum())
for l in range(1,19):
    a=Dm[valid,l]; prior=cum[valid,l-1]; f=fail_[valid]
    ms,mf=a[~f].mean(),a[f].mean()
    pop=(ms*Ns+mf*Nf)/(Ns+Nf)
    r_=np.corrcoef(a,prior)[0,1]
    after=cum[valid,l]
    tw=(prior>0)&(after<=0); tr=(prior<=0)&(after>0)
    tw_w=tw[~f].sum()*wS+tw[f].sum()*wF; tr_w=tr[~f].sum()*wS+tr[f].sum()*wF
    print(l,'%.2f %.2f %.2f %.2f'%(ms,mf,pop,r_),int(tw.sum()),int(tr.sum()),round(float((tr_w-tw_w)/(Ns+Nf)*1000),1))
```