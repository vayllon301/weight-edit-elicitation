Something does change at the second global layer. Up to hidden state 11, math and non-math tokens have the same residual norm outside the massive channels, within 3%. Right after layer 11 (the global layer) the math tokens' norm is 35% larger, 47% larger after layer 12, and the gap then shrinks to 6% by the last layer.

Unlike RMU, though, the excess does not lie along a common mean direction: the mean-difference share stays near 0.2-0.27 at every layer and the math tokens' cosine with it is flat at 0.13-0.21. There is also an obvious confound. Math answer tokens sit at positions 90-260, while the short non-math texts end by about position 110. A global attention layer could produce larger outputs further into the context for purely positional reasons.

So I'll split the norms at hidden states 11, 12 and 13 by position bin for both groups. Within math texts I'll also separate digit tokens from other tokens, and I'll list which channels carry the extra squared norm.
- A content effect would show math tokens above non-math tokens at matched positions, possibly concentrated on digits or a few channels. That would make layer 11 a causal target.
- A purely positional effect would show both groups rising together with position.

```python
toks2=[[ts(x) for x in np.load('/work/fwd_2/input_ids.npy', mmap_mode='r')[i,:LN[i]]] for i in range(len(LN))]
for l in [11,12,13]:
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    allX=[];
    rows=[]
    for i in range(H.shape[0]):
        X=H[i,PLall[i]:LN[i]].astype(np.float32); pos=np.arange(PLall[i],LN[i])
        dig=np.array([toks2[i][p].replace('\u2581','').isdigit() for p in pos])
        rows.append((i<nm, X, pos, dig))
    Xall=np.concatenate([r[1] for r in rows]); Xall=Xall[np.isfinite(Xall).all(1)]
    keep=np.argsort(-np.abs(Xall).mean(0))[8:]
    bins=[(0,60),(60,90),(90,120),(120,180),(180,330)]
    out=[]
    for (a,b) in bins:
        for grp in (True,False):
            for dg in ([False,True] if grp else [False]):
                v=[np.linalg.norm(X[k][keep]) for (g,X,pos,dig) in rows if g==grp for k in range(len(pos)) if a<=pos[k]<b and dig[k]==dg and np.isfinite(X[k]).all()]
                if len(v)>20: out.append(f"{'M' if grp else 'N'}{'d' if dg else ''}[{a},{b}):{np.mean(v):.0f}({len(v)})")
    msq_m=np.mean(np.concatenate([X[np.isfinite(X).all(1)][:,keep]**2 for (g,X,pos,dig) in rows if g]),0)
    msq_n=np.mean(np.concatenate([X[np.isfinite(X).all(1)][:,keep]**2 for (g,X,pos,dig) in rows if not g]),0)
    ex=msq_m-msq_n; top=np.argsort(-ex)[:5]
    print(l, ' '.join(out)); print('   excess top ch', [(int(keep[j]), round(float(ex[j]/ex.sum()),2)) for j in top], 'tot share>0', round(float(ex[ex>0].sum()/msq_m.sum()),2))
```