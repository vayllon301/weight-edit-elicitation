The norm and reservoir picture is striking. Final-norm weights have a mean of 25.4 and reach 300. Coordinate 163 is enormous, up to about 50,000 at the last layer, so it dominates the final RMS of about 1,332. That fits the reservoir I recorded earlier, and I already closed its correct-vs-wrong differential. Overflows in this sample occur only at coordinate 163, so I can drop that one coordinate from the projections and still keep it in the RMS where finite.

Now for the validation. If the reconstructed gaps match the tool's reported logprob gaps, I'll know the norm's parameterisation (`w` or `1+w`) and the ledger will be trustworthy. If they don't match, I'd need to fix the formula before reading any attribution. After that I'll compute the per-layer ledger for failures and successes, split by digit class.

```python
import numpy as np, json
d7='/work/fwd_7/'
Hs=np.load('/work/tf7_Hs.npy',mmap_mode='r'); idx=np.load('/work/tf7_idx.npy')
TI7=np.load(d7+'topk_ids.npy',mmap_mode='r'); TL7=np.load(d7+'topk_logprobs.npy',mmap_mode='r')
ii=R7[:,0].astype(int)[idx]; tt=R7[:,1].astype(int)[idx]-1
c=R7[idx,2].astype(int); isfail=(R7[idx,4]==0)
digset=set(digit_ids)
tk=np.array([TI7[i,t] for i,t in zip(ii,tt)]); tv=np.array([TL7[i,t] for i,t in zip(ii,tt)])
w=np.zeros(len(idx),int); gap=np.full(len(idx),np.nan)
for n in range(len(idx)):
    a=tk[n]; 
    if isfail[n]:
        w[n]=a[0]
        p=np.where(a==c[n])[0]
    else:
        alt=[k for k in range(5) if a[k]!=c[n] and int(a[k]) in digset]
        w[n]=a[alt[0]] if alt else -1
        p=np.array([alt[0]]) if alt else np.array([])
    if len(p):
        gap[n]=(tv[n,0]-tv[n,p[0]]) if isfail[n] else (tv[n,0]-tv[n,p[0]])
print('fails',isfail.sum(),'succ',(~isfail).sum(),'succ w/ digit alt',(w>=0).sum()-isfail.sum())
h18=np.array(Hs[18]); ok=~np.isinf(h18).any(1)
rms=np.sqrt(np.mean(np.where(np.isinf(h18),0,h18)**2,axis=1)+1e-6)
sub=np.where(ok&(w>=0)&~np.isnan(gap))[0]
for nm,G in [('1+w',1+gw),('w',gw)]:
    x=h18[sub]*G/rms[sub,None]
    lg=np.einsum('ij,ij->i',E[c[sub]]-E[w[sub]],x)
    # signed: fails should give c<w => lg negative = -gap; succ positive = +gap
    tgt=np.where(isfail[sub],-gap[sub],gap[sub])
    print(nm,'corr %.4f'%np.corrcoef(lg,tgt)[0,1],'slope %.3f'%(np.dot(lg,tgt)/np.dot(lg,lg)),'med abs err %.3f'%np.median(np.abs(lg-tgt)))
# reservoir share
print('c163 at last layer: med %.0f, share of final energy med %.3f'%(np.median(h18[ok,163]),np.median(h18[ok,163]**2/np.sum(h18[ok]**2,axis=1))))
top=np.argsort(-np.abs(gw))[:6]; print('top final gains',[(int(k),round(float(gw[k]),1)) for k in top],'gw at163',float(gw[163]),'E col163 rms',float(E[:2000,163].std()))
```