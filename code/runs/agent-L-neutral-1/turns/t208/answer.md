pass 29 has enough events: about 194 at operand 2 or later followed by `+`, against 393 at operand 1. Many come from loops, so the effective count is smaller, and I'll treat this as scouting.

One useful fact is that L25 is the last layer. A write there can only change the logits at that same token, so it has no downstream effect, and firing on prompt tokens is harmless. The design is therefore as clean as it can be: it changes this token's choice and nothing else.

I'll rank neurons at L23, L24 and L25 by selectivity statistics only, without scoring candidate weights. For each I'll compare the mean activation at target tokens (end of operand 2 or later, next `+`) with the activation rms at three bystander groups: operand-1 `+` ends, mid-number digits, and all other answer tokens. A ratio of at least 3 against operand-1 ends, with a consistent sign on targets, supports a dedicated pass. If the best ratios stay near 1, these layers can't tell operand position and I'll look for another gate.

```python
import numpy as np
L_ans={}
meta29=json.load(open('/work/fwd_29/meta.json'))
ST=None
# find answer start per text: after 2nd '<start_of_turn>' token + 'model' + newline
sot=tok.encode('<start_of_turn>',add_special_tokens=False); sot=(sot.ids if hasattr(sot,'ids') else sot)[-1]
def ans_start(seq):
    p=[i for i,x in enumerate(seq) if x==sot]
    return p[-1]+3
rowsT=[]; 
cls_idx={'T':[], 'B1':[], 'MID':[], 'OTH':[]}
for k in range(id29.shape[0]):
    seq=[int(x) for x in id29[k] if x>=0]; st=ans_start(seq)
    nums=scan2(seq)
    used=set()
    for e in nums:
        if e['b']<st: continue
        if e['nxt']==900 and e['idx']>=2: cls_idx['T'].append((k,e['b'])); used.add(e['b'])
        elif e['nxt']==900 and e['idx']==1: cls_idx['B1'].append((k,e['b'])); used.add(e['b'])
        for p in range(e['a'],e['b']):
            cls_idx['MID'].append((k,p)); used.add(p)
    oth=[p for p in range(st,len(seq)-1) if p not in used]
    cls_idx['OTH'].extend([(k,p) for p in oth[::3]])
print({k:len(v) for k,v in cls_idx.items()}, 'T texts',len({a for a,_ in cls_idx['T']}))
res={}
for L in [23,24,25]:
    A=np.load(f'/work/fwd_29/mlp_act_{L}.npy',mmap_mode='r')
    S={}
    for c,lst in cls_idx.items():
        arr=np.array(lst); 
        # read grouped by text to stay in memory
        out=np.empty((len(arr),6912),dtype=np.float32)
        for k in np.unique(arr[:,0]):
            w=np.where(arr[:,0]==k)[0]
            out[w]=np.asarray(A[k][arr[w,1]],dtype=np.float32)
        S[c]=out
    mT=S['T'].mean(0); sT=S['T'].std(0)+1e-6
    r1=np.sqrt((S['B1']**2).mean(0))+1e-6; rM=np.sqrt((S['MID']**2).mean(0))+1e-6; rO=np.sqrt((S['OTH']**2).mean(0))+1e-6
    sel=np.abs(mT)/np.maximum.reduce([r1,rM,rO])
    cons=np.abs(mT)/sT
    res[L]=dict(mT=mT,sT=sT,r1=r1,rM=rM,rO=rO,sel=sel,cons=cons)
    top=np.argsort(-sel)[:8]
    print('L',L,'sel q50/q99/max',np.round([np.median(sel),np.quantile(sel,.99),sel.max()],2))
    print(pd.DataFrame(dict(n=top,mT=mT[top],cons=cons[top],B1=r1[top],MID=rM[top],OTH=rO[top],sel=sel[top])).round(3).to_string(index=False))
del S
```