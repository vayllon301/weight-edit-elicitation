The baseline's done. I'll now write the paired screening function: it compares any later pass against this baseline, text by text, on each measure, with standard errors.

Then I'll build the first diagnostic. In layers 12-24, I scale how strongly each layer writes into dimension 1038, multiplying (1 + weight) by 0.8 in both post-norms. That lowers the register by about 10% at the late layers, raising the effective gain of everything downstream by about 11%, without touching anything else directly.

If gain matters, this should move reference log probability, especially on digits, and the loop-entry measures. I'll run the opposite sign next, because a real operating-point effect should reverse sign. If neither direction improves reference scores, I'll close the line.

```python
import numpy as np, json, torch
idx2=json.load(open('/work/tf_scr2_index.json'))
digit_ids=set(tok.encode(str(d),add_special_tokens=False).ids[-1] for d in range(10))
print('digit ids',sorted(digit_ids))
def load_lp(k):
    return np.load(f'/work/fwd_{k}/token_logprobs.npy',mmap_mode='r'),np.load(f'/work/fwd_{k}/input_ids.npy',mmap_mode='r'),np.array(json.load(open(f'/work/fwd_{k}/meta.json'))['lengths'])
LP0,I0,N0=load_lp(15)
def metrics(LPx,Ix,Nx):
    m={k:[] for k in ['R_all','R_dig','C','W','E0','E1','E2','E_all']}
    for i,r in enumerate(idx2):
        n=int(Nx[i]); st=r['n_pre']; lp=np.asarray(LPx[i,st-1:n-1],dtype=np.float64); ids=np.asarray(Ix[i,st:n])
        if r['group']=='R':
            m['R_all'].append((i,lp.mean())); d=np.isin(ids,list(digit_ids))
            if d.sum()>0: m['R_dig'].append((i,lp[d].mean()))
        elif r['group'] in 'CW':
            m[r['group']].append((i,lp.mean()))
        else:
            m['E_all'].append((i,lp.mean()))
            for q,(a,b) in enumerate(r['spans'][:3]):
                m[f'E{q}'].append((i,float(np.asarray(LPx[i,a-1:b],dtype=np.float64).sum())))
    return {k:dict(v) for k,v in m.items()}
BASE=metrics(LP0,I0,N0)
print({k:(len(v),round(float(np.mean(list(v.values()))),3)) for k,v in BASE.items()})
def screen(k,label=''):
    L,I_,N=load_lp(k); M=metrics(L,I_,N); out={}
    for key,v in M.items():
        ids=[i for i in v if i in BASE[key]]; d=np.array([v[i]-BASE[key][i] for i in ids])
        out[key]=(round(float(d.mean()),4),round(float(d.std()/np.sqrt(len(d))),4))
    print(label,out); return out
c=0.8; layers=list(range(12,25))
def set_gain(c,layers):
    for l in layers:
        for nm in ['post_attention_layernorm','post_feedforward_layernorm']:
            n=f'model.layers.{l}.{nm}.weight'; w=load_original(n).float().clone()
            w[1038]=(1+w[1038])*c-1
            save_edit(n,w)
set_gain(c,layers)
print(len(edited()),'tensors edited; sample',round(float(load('model.layers.12.post_feedforward_layernorm.weight')[1038]),2),'was',round(float(load_original('model.layers.12.post_feedforward_layernorm.weight')[1038]),2))
open('/work/notes.md','a').write(f"""
Pass 16 (teacher-forced, EDITED, tf_scr2_texts.json): EDIT = post_attention_layernorm and post_feedforward_layernorm weight[1038], (1+w)*{c}, layers 12-24 (register gain diagnostic). Score against pass 15 with screen(). Compare opposite sign next.
""")
```