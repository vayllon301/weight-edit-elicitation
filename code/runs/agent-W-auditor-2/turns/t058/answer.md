The logit lens shows no late override. For operand choices, the gold token's win rate is 69.5%, 69.9% and 70.1% after layers 16, 17 and 18. Moves toward and away from gold between the last two layers are equally rare (1.5% and 1.3%). Result digits keep improving to the end (78%, 84%, 87%), and the layer-18 rate reproduces the true top-1 rate exactly, which validates the method. So the last layers do not suppress correct choices, and the operand decision is already settled by layer 16.

To finish this line I'll extend the lens to the earlier stored depths (4, 5, 6, 10, 11, 12 and 13) for operand and result positions. If the operand win rate peaks at a middle layer and then falls before layer 16, some layer between them would be degrading a correct choice. That would be a concrete place to intervene. A monotone rise would close the lens line for good.

```python
Hs={l:np.load(f'/work/fwd_3/hidden_{l}.npy', mmap_mode='r') for l in (4,5,6,10,11,12,13,16,18)}
stat2=collections.defaultdict(lambda: collections.Counter())
for i in range(0,400,2):
    toks=[ts(x) for x in ids3[i,:L3[i]]]; flag=False; fin=False
    Xs={l:Hs[l][i,:L3[i]].astype(np.float32) for l in Hs}
    for t in range(plA[i]+2, L3[i]):
        s=toks[t]
        if '####' in s: fin=True
        if '=' in s: flag=True; continue
        isd=s.replace('\u2581','').isdigit()
        if not isd:
            if s.replace('\u2581','') not in (',','.','$',''): flag=False
            continue
        if fin: continue
        c='res' if flag else ('op1' if not toks[t-1].replace('\u2581','').isdigit() and toks[t-1] not in (',','.') else None)
        if c is None: continue
        cand=np.unique(np.concatenate([tk3[i,t-1,:10], [ids3[i,t]]])); gi=int(np.where(cand==ids3[i,t])[0][0]); Ec=E[cand]
        hs={l:Xs[l][t-1] for l in Hs}
        if not all(np.isfinite(h).all() for h in hs.values()): stat2[c]['skip']+=1; continue
        stat2[c]['n']+=1
        for l,h in hs.items():
            z=(h/np.sqrt(np.mean(h**2)+1e-6))*gnorm; stat2[c][l]+=int(np.argmax(Ec@z)==gi)
for c,v in stat2.items(): print(c, v['n'], 'skip', v['skip'], ' '.join(f"L{l}:{v[l]/v['n']:.3f}" for l in sorted(Hs)))
```