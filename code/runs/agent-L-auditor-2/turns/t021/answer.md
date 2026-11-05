The pass ran fine. First come the two readouts. For health, I'll report mean logprob and top-1 over answer tokens in each group, plus top-1 on the final key word for facts and logic. For localization, I'll compute at each layer how far apart the group means are at the last prompt token, relative to the within-group spread, and how much the between-group vector changes from one layer to the next. Only the last-prompt row and per-text answer means are loaded, so memory stays small. An isolated spike in layer-to-layer change would be a location to test, while a smooth curve would argue against a sharp math gate.

```python
import numpy as np, json, pandas as pd
d='/work/fwd_3/'
ix3=json.load(open('/work/tf_contrast_index.json')); tx3=json.load(open('/work/tf_contrast_texts.json'))
meta3=json.load(open(d+'meta.json')); L3=meta3['lengths']
ids3=np.load(d+'input_ids.npy',mmap_mode='r'); lp3=np.load(d+'token_logprobs.npy',mmap_mode='r'); tk3=np.load(d+'topk_ids.npy',mmap_mode='r')
grp=np.array([r['group'] for r in ix3])
# health
hl={}
for i,r in enumerate(ix3):
    n=L3[i]; np_=r['n_pre']
    if n<=np_: continue
    x=np.array(ids3[i,:n]); l=np.array(lp3[i,:n-1]); t=np.array(tk3[i,:n-1,0])
    b=np.arange(np_,n)
    s=hl.setdefault(r['group'],dict(lp=[],top=[],keytop=[],keylp=[]))
    s['lp'].append(l[b-1].mean()); s['top'].append((t[b-1]==x[b]).mean())
    if 'key' in r:
        # last word tokens: last few tokens excluding final punctuation
        kids=tok.encode(' '+r['key'],add_special_tokens=False).ids
        e=n-1 if x[n-1]==tok.encode('.',add_special_tokens=False).ids[0] else n
        kp=np.arange(e-len(kids),e)
        s['keytop'].append(float((t[kp-1]==x[kp]).all())); s['keylp'].append(l[kp-1].sum())
print({g:{k:round(float(np.mean(v)),3) for k,v in s.items() if len(v)} for g,s in hl.items()})
# localization at last prompt token
last=np.array([r['n_pre']-1 for r in ix3])
res=[]; deltas={}
for lay in range(27):
    H=np.load(d+f'hidden_{lay}.npy',mmap_mode='r')
    X=np.stack([np.asarray(H[i,last[i]],dtype=np.float32) for i in range(len(ix3))])
    M=np.isin(grp,['M_correct','M_wrong'])
    g={k:X[grp==k] for k in ['T_summ','F_fact','R_logic']}; g['M']=X[M]
    mu={k:v.mean(0) for k,v in g.items()}
    within=np.mean([((v-mu[k])**2).sum(1).mean() for k,v in g.items()])
    dv={f'M-{k}':mu['M']-mu[k] for k in ['T_summ','F_fact','R_logic']}
    deltas[lay]=dv
    rec=dict(l=lay,norm=round(float(np.linalg.norm(X,axis=1).mean()),1),within=round(float(np.sqrt(within)),2))
    for k,v in dv.items():
        rec[k]=round(float(np.linalg.norm(v)/np.sqrt(within)),2)
    res.append(rec)
df=pd.DataFrame(res)
# step change between layers, relative to the size of the vector
for k in ['M-T_summ','M-F_fact','M-R_logic']:
    st=[np.nan]+[float(np.linalg.norm(deltas[l][k]-deltas[l-1][k])/(np.linalg.norm(deltas[l][k])+1e-6)) for l in range(1,27)]
    df['step_'+k[2:6]]=np.round(st,2)
print(df.to_string(index=False))
```