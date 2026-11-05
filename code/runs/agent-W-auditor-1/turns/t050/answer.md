Pass 2 has what I need: 1,016 texts, each position with the model's top 5. Reference solutions use a plain style (`4*5 = 20`) unlike the model's own, so failures at ordinary word tokens will mostly be style mismatch and say little. The clean failures are digit-on-digit confusions, where the correct next token is a digit and the model's top token is a different digit: an arithmetic or number-reading error at that point, with the context held fixed.

I'll count those positions among the 800 reference texts, split between result digits (right after `=`) and others. I'll also look at the rank of the correct digit when it misses, and the logit gap. If the correct digit is usually the close runner-up, the failures are near-ties. I'll also pick a text subset within the 6 GB limit for all-layer hidden states. A few hundred failing digit positions would be enough for the ledger.

```python
import numpy as np, json
d2='/work/fwd_2/'
ids2=np.load(d2+'input_ids.npy',mmap_mode='r'); lp=np.load(d2+'token_logprobs.npy',mmap_mode='r')
ti=np.load(d2+'topk_ids.npy',mmap_mode='r'); tl=np.load(d2+'topk_logprobs.npy',mmap_mode='r')
m2=json.load(open(d2+'meta.json')); LEN=np.array(m2['lengths'])
N=800
digit_ids={}
for c in '0123456789':
    e=tok.encode(c); ii=e.ids if hasattr(e,'ids') else e
    digit_ids[int(ii[-1])]=c
print(digit_ids)
# alignment check: topk index t predicts token t+1 ?
X=np.array(ids2[:N]); TI=np.array(ti[:N,:,0]); LP=np.array(lp[:N]); TL=np.array(tl[:N])
isd=np.isin(X,list(digit_ids))
recs=[]
for i in range(N):
    L=LEN[i]; x=X[i,:L]
    sot=np.where(x==105)[0]; m=sot[-1]+3
    prev_txt=None
    for t in range(m,L):
        if not isd[i,t]: continue
        pred=TI[i,t-1]; ok=pred==x[t]
        # context class: is previous non-space token '=' ?
        pv=tok.decode([int(x[t-1])]); pv2=tok.decode([int(x[t-2])])
        first_after_eq = ('=' in pv) or (pv.strip()=='' and '=' in pv2)
        indigit=int(x[t-1]) in digit_ids
        recs.append((i,t,int(x[t]),int(pred),bool(ok),bool(first_after_eq),bool(indigit),float(np.exp(0)*0+TL[i,t-1,0]),float(LP[i,t-1]),int(pred in digit_ids)))
R=np.array(recs,dtype=float)
print('digit positions',len(R),'top1 right %.3f'%R[:,4].mean())
for nm,mk in [('first digit after =',R[:,5]==1),('later digit after digit',R[:,6]==1),('other first digit',(R[:,5]==0)&(R[:,6]==0))]:
    s=R[mk]; bad=s[(s[:,4]==0)&(s[:,9]==1)]
    print(nm,len(s),'acc %.3f'%s[:,4].mean(),'digit-on-digit fails',len(bad),'median gap %.2f'%np.median(bad[:,7]-bad[:,8]) if len(bad) else '')
# rank of right token among top5 at digit-on-digit fails
TK=np.array(ti[:N])
bad=R[(R[:,4]==0)&(R[:,9]==1)]
rk=[]
for row in bad:
    i,t=int(row[0]),int(row[1]); a=TK[i,t-1]; w=np.where(a==X[i,t])[0]; rk.append(int(w[0]) if len(w) else 9)
print('rank dist',np.bincount(rk,minlength=10))
for cap in (200,220,240,260,280):
    sel=np.where(LEN[:N]<=cap)[0]
    print(cap,'texts',len(sel),'GB %.2f'%(len(sel)*cap*19*640*2/1e9),'fails kept',int(np.isin(bad[:,0],sel).sum()))
```