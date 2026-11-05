Now I'll apply the fork readout: the mean margin shift, with standard errors clustered by problem, and how many forks changed side. Fork groups are all split by source, so the 172 forks independent of P1 are read apart from those built from its own arms. I'm adding a reference-screen reader for the 438 overlap texts, which should roughly reproduce P1's known result and so validate the readout itself.

```python
import numpy as np, pandas as pd
pid=np.array([f['id'] for f in info]); tt=np.array([f['t'] for f in info]); crit=np.array([f['crit'] for f in info])
sNoP=np.array([f['src_noP'] for f in info]); sAll=np.array([f['src_all'] for f in info])
def cl_mean(x,g):
    u,inv=np.unique(g,return_inverse=True); s=np.bincount(inv,weights=x); n=np.bincount(inv).astype(float); mu=s.sum()/n.sum()
    return mu,np.sqrt(((s-mu*n)**2).sum())/n.sum()
def fork_tab(folder,base=(la0,lb0),groups=None):
    la,lb=fork_read(folder); m=la-lb; mb=base[0]-base[1]; dm=m-mb
    G=groups or {'noP t>0':sNoP&(tt>0),'noP crit':sNoP&(tt>0)&crit,'noP noncrit':sNoP&(tt>0)&~crit,'noP opener':sNoP&(tt==0),'P-only t>0':~sNoP&(tt>0),'P-only crit':~sNoP&(tt>0)&crit,'all t>0':tt>0}
    rows=[]
    for k,msk in G.items():
        mu,se=cl_mean(dm[msk],pid[msk])
        rows.append((k,int(msk.sum()),len(set(pid[msk])),round(mu,3),round(se,3),round(mu/se,2) if se>0 else 0,round(dm[msk].std(),3),int(((mb[msk]<=0)&(m[msk]>0)).sum()),int(((mb[msk]>0)&(m[msk]<=0)).sum()),round((mb[msk]>0).mean(),3),round((m[msk]>0).mean(),3)))
    return pd.DataFrame(rows,columns=['group','forks','probs','dMargin','se','z','sd','toRight','toWrong','pick0','pick1'])
def ref_sub(folder):
    lp=np.load(folder+'token_logprobs.npy',mmap_mode='r'); tp=np.load(folder+'topk_ids.npy',mmap_mode='r')
    keep=B_<nref; bb=B_[keep]; tq=Tt[keep]
    l=np.asarray(lp[bb+nf,tq]).astype(np.float64); t1=np.asarray(tp[bb+nf,tq,0])==Tid[keep]
    out=[]
    for g,msk in list(GR.items())+[('ALL',pd.Series(True,index=T3.index))]:
        m_=msk.values[keep]; b2=bb[m_]; nb=np.bincount(b2,minlength=nref).astype(float)
        dt=t1[m_].astype(float)-base_top[keep][m_].astype(float); dl=l[m_]-base_lp[keep][m_]
        def cl(d):
            s=np.bincount(b2,weights=d,minlength=nref); mu=s.sum()/nb.sum(); return mu,np.sqrt(((s-mu*nb)**2).sum())/nb.sum()
        a,b=cl(dt); c,d=cl(dl); out.append((g,int(m_.sum()),round(100*a,2),round(100*b,2),round(c,3),round(d,3)))
    return pd.DataFrame(out,columns=['group','n','dTop1','se','dLogp','se_lp'])
print(fork_tab('/work/fwd_24/').to_string())
print(ref_sub('/work/fwd_24/').to_string())
```