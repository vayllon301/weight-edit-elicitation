The manipulation pass is done. I'll compute the same per-head statistics as for pass 6, side by side, and compare the answer-token log-probability on these two texts. The check passes if effective positions fall in the reading heads and BOS share stays near its earlier level. Two texts can only verify the mechanism, so the log-probability change here says nothing about gain.

```python
def attn_stats(k):
    d=f'/work/fwd_{k}/'
    meta=json.load(open(d+'meta.json')); L=meta['lengths']
    out={}
    for l in (5,11,17):
        Af=np.asarray(np.load(d+f'attn_{l}.npy',mmap_mode='r')).astype(np.float64)
        rows=[]
        for h in range(Af.shape[1]):
            vals=[]
            for r in range(Af.shape[0]):
                T=L[r]
                for t in range(int(T*0.4),T):
                    p=Af[r,h,t,:t+1].copy(); bos=p[0]; rest=p[1:]; tot=rest.sum()
                    if tot<=1e-6: continue
                    q=rest/tot; H=-(q[q>0]*np.log(q[q>0])).sum()
                    vals.append((bos,q.max(),np.exp(H)))
            v=np.array(vals)
            rows.append((round(float(v[:,0].mean()),3),round(float(np.median(v[:,1])),3),round(float(np.median(v[:,2])),1)))
        out[l]=rows
    return out
a6,a23=attn_stats(6),attn_stats(23)
print('head: BOS share, top share, effective positions  (pass 6 -> K0)')
for l in (5,11,17):
    for h in range(4):
        print('L%d h%d'%(l,h),a6[l][h],'->',a23[l][h])
lp6=np.load('/work/fwd_6/token_logprobs.npy').astype(np.float64); lp23=np.load('/work/fwd_23/token_logprobs.npy').astype(np.float64)
L=json.load(open('/work/fwd_6/meta.json'))['lengths']
for r in range(2):
    T=L[r]; s=int(T*0.4)
    print('text',r,'mean logprob last 60 pct, orig %.3f K0 %.3f'%(np.nanmean(lp6[r,s:T-1]),np.nanmean(lp23[r,s:T-1])))
```