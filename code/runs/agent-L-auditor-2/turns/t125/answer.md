This settles the duplicate groups, and it also explains an older result:

- **L10, L11, L16:** the groups are negligible, carrying 1-3% of the layer's mean write. Nothing to pursue.
- **L25:** the 54 neurons are **near-constant bias units**. They have a coefficient of variation of 0.13 (0.97 is typical), and activations are far above normal, up to 7.9. They produce most of L25's mean write (2.4 of 4.4, cosine 0.88). So they add a static logit bias at every token, the same bias whatever the context.

That accounts for the strange "repetition brake" from passes 4 and 5. The top `up_proj` directions in L25 are most likely the read side of these static units, so removing them wiped out the output bias and broke generation. It was the model's static output bias, and the "brake" label overstated what I knew. A static bias can't be selective for math either, and changing it would be the global output change the task rules out. This line is closed.

So far I've judged every edit on all reference tokens and on digits, but those mix very different things. Failures happen at set-up, and arithmetic is fine. A digit that *starts* an equation (choosing what to compute) measures reasoning. A digit after `=` mostly measures sums. This matters because the pass 22 reference gain may have been hidden or inflated by that mix.

I can re-score every teacher-forced pass I already have, with no new pass, by splitting reference tokens into operand digits, result digits, operators and other tokens. I'm looking for any edit with a real positive slope on operand digits. That would give me the planning-specific signal I've lacked. If nothing shows up, the existing edits simply don't touch planning.

```python
import numpy as np, re
print(inspect.getsource(load_lp)[:500])
L0,I0,N0=load_lp(15)
ri=[i for i,r in enumerate(idx2) if r['group']=='R']
ex=ri[0]; st=idx2[ex]['n_pre']; print(repr(TK.decode([int(t) for t in I0[ex,st:min(st+45,int(N0[ex]))]],skip_special_tokens=False)[:170]))
dig=set(int(i) for i in digit_ids)
piece={}
def pc_(t):
    t=int(t)
    if t not in piece: piece[t]=TK.decode([t],skip_special_tokens=False)
    return piece[t]
OPS={'+','-','*','/','x','×','÷','=','$','%'}
Cat={}
for i in ri:
    n=int(N0[i]); st=idx2[i]['n_pre']; ids=I0[i,st:n]; c=np.zeros(len(ids),np.int8); after=False; 
    for j,t in enumerate(ids):
        p=pc_(t); s=p.strip()
        if '=' in p: after=True; c[j]=2; continue
        if int(t) in dig:
            c[j]=3 if after else 1
        else:
            if s in OPS or s in('+','-','*','/'): c[j]=2
            if s!='' and s not in(',','.') and not(int(t) in dig): 
                if any(ch.isalpha() for ch in s) or '\n' in p: after=False
            if '\n' in p: after=False
    Cat[i]=c
names={0:'other',1:'operand_dig',2:'operator',3:'result_dig'}
tot=np.bincount(np.concatenate(list(Cat.values())),minlength=4); print('token counts',{names[k]:int(tot[k]) for k in range(4)})
def split(k):
    L,I_,N=load_lp(k); o={}
    for c in range(4):
        v={}
        for i in ri:
            n=int(N[i]); st=idx2[i]['n_pre']; lp=np.asarray(L[i,st-1:n-1],dtype=np.float64); m_=Cat[i]==c
            if m_.sum()>0: v[i]=lp[m_].mean()
        o[c]=v
    return o
B0=split(15); print('baseline',{names[c]:round(float(np.mean(list(B0[c].values()))),3) for c in range(4)})
lab={16:'reg x.8',17:'B x.5',18:'B x1.4',19:'sink0',21:'L17 x.5',22:'L5/11/17 x.5'}
for k,nm in lab.items():
    S=split(k); r={}
    for c in range(4):
        ids_=[i for i in S[c] if i in B0[c]]; d=np.array([S[c][i]-B0[c][i] for i in ids_]); r[names[c]]=(round(float(d.mean()),4),round(float(d.std()/np.sqrt(len(d))),4))
    print(nm,r)
```