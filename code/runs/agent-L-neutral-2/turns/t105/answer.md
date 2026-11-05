That's the code I needed. I'll turn its walker into a fixed index mapping each of the 95,187 rows to its text position, so token classes stay the same across all screening passes. The screen will report the change in top-1 and log-probability for each class, with standard errors clustered by text, since tokens in one solution aren't independent. It also counts gained and lost decisions.

Two checks must pass before I trust it. The index has to reproduce the stored log-probabilities exactly, and comparing pass 3 with itself has to give zero everywhere. If either fails, the alignment is wrong and I won't spend the pass.

```python
import numpy as np, json, pandas as pd
assert 'dig' in globals(); 
ii3=np.load('/work/fwd_3/input_ids.npy',mmap_mode='r'); m3=json.load(open('/work/fwd_3/meta.json')); lens3=m3['lengths']
B_=[];Tt=[];Tid=[]
for b in range(1024):
    n=int(lens3[b]); ids_=np.array(ii3[b,:n]); a=int(np.where(ids_==105)[0][-1])+3
    strs=[tok.decode([int(q)],skip_special_tokens=False) for q in ids_[a:]]
    j=0
    while j<len(strs):
        tid=int(ids_[a+j])
        if tid in dig:
            k=j
            while k+1<len(strs) and (int(ids_[a+k+1]) in dig or (strs[k+1] in (',','.') and k+2<len(strs) and int(ids_[a+k+2]) in dig)): k+=1
            for q in range(j,k+1):
                if int(ids_[a+q]) in dig: B_.append(b);Tt.append(a+q-1);Tid.append(int(ids_[a+q]))
            j=k+1; continue
        B_.append(b);Tt.append(a+j-1);Tid.append(tid); j+=1
B_=np.array(B_);Tt=np.array(Tt);Tid=np.array(Tid); print('index',len(B_),'expected',len(T3))
base_lp=np.asarray(np.load('/work/fwd_3/token_logprobs.npy',mmap_mode='r')[B_,Tt]).astype(np.float64)
base_top=np.asarray(np.load('/work/fwd_3/topk_ids.npy',mmap_mode='r')[B_,Tt,0])==Tid
print('max |lp-T3.lp| %.6f  top1 mismatch %d'%(np.abs(base_lp-T3.lp.values).max(),int((base_top!=T3.top1.values).sum())))
GR={'copy first':(T3.cls=='copy')&(T3.pos=='first'),'reuse first':(T3.cls=='reuse')&(T3.pos=='first'),'result first':(T3.cls=='result')&(T3.pos=='first'),'novel first':(T3.cls=='novel')&(T3.pos=='first'),'digits later':T3.pos=='later','op':T3.cls=='op','text':T3.cls=='text','eos':T3.cls=='eos'}
def screen(folder):
    lp=np.asarray(np.load(folder+'token_logprobs.npy',mmap_mode='r')[B_,Tt]).astype(np.float64); top=np.asarray(np.load(folder+'topk_ids.npy',mmap_mode='r')[B_,Tt,0])==Tid
    out=[]
    for g,msk in list(GR.items())+[('ALL',pd.Series(True,index=T3.index))]:
        m_=msk.values; bb=B_[m_]; nb=np.bincount(bb,minlength=1024).astype(float)
        dt=(top[m_].astype(float)-base_top[m_].astype(float)); dl=lp[m_]-base_lp[m_]
        def cl(d):
            s=np.bincount(bb,weights=d,minlength=1024); mu=s.sum()/nb.sum(); return mu,np.sqrt(((s-mu*nb)**2).sum())/nb.sum()
        a_,b__=cl(dt); c_,d_=cl(dl)
        out.append((g,int(m_.sum()),base_top[m_].mean(),100*a_,100*b__,int((dt>0).sum()),int((dt<0).sum()),c_,d_))
    return pd.DataFrame(out,columns=['group','n','top1','dTop1_pts','se','gain','lost','dLogp','se_lp'])
print(screen('/work/fwd_3/').round(3).to_string())
```