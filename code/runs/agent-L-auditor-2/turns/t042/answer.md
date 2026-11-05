There is a real signal. Split-half cosines are 0.45–0.65 at every depth, against a null of about 0, so a reproducible direction does separate the model's correct answers from its wrong ones even after this control. Three caveats stop me from treating it as a "reasoning" direction yet:

1. It could be plain text composition: correct answers may simply contain a different mix of tokens, such as more final-answer lines or more digits.
2. It could be a length or position effect, since correct answers are shorter.
3. It has a positive cosine with own-minus-reference style, so it partly measures style.

Only a state-like signal would be worth steering with. So I'll test it out of sample: build the direction from half the problems, then on the other half compare correct and wrong answers **within the same token identity**, and by position in the answer. If the difference holds within identical tokens and appears from the first tokens, it's a state or mode. If it disappears within tokens or only shows up at the end of the answer, it's composition, and writing it as a constant would achieve nothing useful.

```python
import numpy as np, json, pandas as pd
rng=np.random.default_rng(7)
pa=np.array(pairs); own_idx=pa[:,0]; ref_idx=pa[:,1]
perm=rng.permutation(len(pa)); hA=perm[:len(pa)//2]; hB=perm[len(pa)//2:]
out=[]
for lay in [16,20,24]:
    M=means[lay]; D=M[own_idx]-M[ref_idx]
    ca=hA[is_c[hA]]; wa=hA[~is_c[hA]]
    u=D[ca].mean(0)-D[wa].mean(0); u=u/np.linalg.norm(u)
    H=np.load(d+f'hidden_{lay}.npy',mmap_mode='r')
    from collections import defaultdict
    cnt={0:defaultdict(lambda:[0,0.0,0.0]),1:defaultdict(lambda:[0,0.0,0.0])}
    pos={0:[[],[],[],[]],1:[[],[],[],[]]}
    allp={0:[],1:[]}; alltok=[]
    for j in hB:
        i=own_idx[j]; c=int(is_c[j])
        h=np.asarray(H[i,npre7[i]:L7[i]],dtype=np.float32); x=np.asarray(ids7[i,npre7[i]:L7[i]])
        p=h@u
        allp[c].append(p.mean())
        for t,v in zip(x,p):
            s=cnt[c][int(t)]; s[0]+=1; s[1]+=float(v); s[2]+=float(v*v)
        n=len(p)
        if n>=20: pos[c][0].append(p[:20].mean())
        if n>=60:
            pos[c][1].append(p[20:n//2].mean()); pos[c][2].append(p[n//2:-20].mean()); pos[c][3].append(p[-20:].mean())
    T=np.mean(allp[1])-np.mean(allp[0]); sdtext=np.sqrt((np.var(allp[1])+np.var(allp[0]))/2)
    toks=[t for t in cnt[1] if cnt[1][t][0]>=25 and cnt[0].get(t,[0])[0]>=25]
    num=0;den=0;tsd=[]
    for t in toks:
        a=cnt[1][t]; b=cnt[0][t]; f=a[0]+b[0]
        ma=a[1]/a[0]; mb=b[1]/b[0]
        va=max(a[2]/a[0]-ma**2,1e-6); vb=max(b[2]/b[0]-mb**2,1e-6)
        num+=f*(ma-mb); den+=f; tsd.append((np.sqrt((va+vb)/2),f))
    within=num/den; tok_sd=sum(s*f for s,f in tsd)/sum(f for s,f in tsd)
    covered=den/sum(sum(v[0] for v in cnt[k].values()) for k in cnt)
    posd=[round(float((np.mean(pos[1][k])-np.mean(pos[0][k]))/(sdtext+1e-6)),2) for k in range(4)]
    out.append(dict(hidden=lay,nB=len(hB),T_diff=round(float(T),1),T_in_textsd=round(float(T/sdtext),2),within_tok_diff=round(float(within),1),within_share=round(float(within/T),2),token_sd=round(float(tok_sd),1),n_tok_types=len(toks),covered=round(covered,2),by_pos_first20_mid_late_last20=posd))
print(pd.DataFrame(out).to_string(index=False))
```