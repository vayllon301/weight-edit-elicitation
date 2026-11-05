With the baseline in hand, I'll compute the contrast properly. Plain "correct minus wrong" compares different problems and is confounded by difficulty, while "reference minus wrong" is confounded by style. A difference-in-differences cancels both: take (own − reference) on problems the model got right, minus (own − reference) on problems it got wrong. Style cancels because the model's own text sits on both sides of the comparison, and the reference for the same problem absorbs difficulty.

The test of whether this is real is reproducibility: I'll split the problems into two random halves many times and compute the cosine between the two halves' direction vectors. A high cosine means a stable direction worth trying as an edit. A cosine near zero means noise and no such mode. I'll also report how large it is relative to token-level variation, since that sets the strength an edit would need.

```python
import numpy as np, json, pandas as pd
d='/work/fwd_7/'
ix7=json.load(open('/work/tf_screen_index.json')); m7=json.load(open(d+'meta.json')); L7=np.array(m7['lengths'])
grp7=np.array([r['group'] for r in ix7]); npre7=np.array([r['n_pre'] for r in ix7]); pid7=np.array([r['id'] for r in ix7])
ids7=np.load(d+'input_ids.npy',mmap_mode='r'); lp7=np.load(d+'token_logprobs.npy',mmap_mode='r')
# baseline answer logprob per text
base_lp=np.array([np.array(lp7[i,npre7[i]-1:L7[i]-1]).mean() for i in range(len(ix7))])
np.save('/work/base_lp7.npy',base_lp)
print({g:round(float(base_lp[grp7==g].mean()),3) for g in 'CWR'})
own=np.where(grp7!='R')[0]; ref=np.where(grp7=='R')[0]
ref_of={pid7[i]:i for i in ref}
pairs=[(i,ref_of[pid7[i]]) for i in own]
is_c=np.array([grp7[i]=='C' for i,_ in pairs])
means={}
for lay in [8,12,16,20,24,26]:
    H=np.load(d+f'hidden_{lay}.npy',mmap_mode='r')
    M=np.zeros((len(ix7),1152),dtype=np.float32)
    for i in range(len(ix7)):
        M[i]=np.asarray(H[i,npre7[i]:L7[i]],dtype=np.float32).mean(0)
    means[lay]=M
np.savez('/work/screen_means.npz',**{str(k):v for k,v in means.items()})
rng=np.random.default_rng(1)
out=[]
for lay,M in means.items():
    D=np.stack([M[a]-M[b] for a,b in pairs])   # own - reference
    did=D[is_c].mean(0)-D[~is_c].mean(0)
    sty=D.mean(0)
    rawcw=M[[a for a,_ in pairs]][is_c].mean(0)-M[[a for a,_ in pairs]][~is_c].mean(0)
    sp=[]
    for r in range(30):
        c=rng.permutation(np.where(is_c)[0]); w=rng.permutation(np.where(~is_c)[0])
        a=D[c[:len(c)//2]].mean(0)-D[w[:len(w)//2]].mean(0); b=D[c[len(c)//2:]].mean(0)-D[w[len(w)//2:]].mean(0)
        sp.append(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    # null: same split with shuffled labels
    nl=[]
    for r in range(30):
        lab=rng.permutation(is_c); c=rng.permutation(np.where(lab)[0]); w=rng.permutation(np.where(~lab)[0])
        a=D[c[:len(c)//2]].mean(0)-D[w[:len(w)//2]].mean(0); b=D[c[len(c)//2:]].mean(0)-D[w[len(w)//2:]].mean(0)
        nl.append(a@b/np.linalg.norm(a)/np.linalg.norm(b))
    sd=D.std(0); 
    out.append(dict(hidden=lay,did_norm=round(float(np.linalg.norm(did)),1),style_norm=round(float(np.linalg.norm(sty)),1),within_rms=round(float(np.sqrt((D-D.mean(0)).__pow__(2).sum(1).mean())),1),
        split_cos=round(float(np.mean(sp)),3),null_cos=round(float(np.mean(nl)),3),cos_did_style=round(float(did@sty/np.linalg.norm(did)/np.linalg.norm(sty)),3),cos_did_rawcw=round(float(did@rawcw/np.linalg.norm(did)/np.linalg.norm(rawcw)),3)))
print(pd.DataFrame(out).to_string(index=False))
```