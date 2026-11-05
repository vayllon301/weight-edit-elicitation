I need to correct something I told you. My count of 673 "on" units used the lower of the two controls, so those units were quiet in at least one control, not in both as I said. The "off" count, by contrast, already compared math against both controls, so the 16-to-1 imbalance was an unfair comparison and it's overstated. With both controls required, S8 has 133 units, not 673, and that set uses a looser cutoff of 0.5.

The closure of the off-units line doesn't depend on that ratio, since 41 units scattered across layers is few either way, but the number I gave you and wrote in the notes is wrong. I'll compute the true mirror count with the same 0.25 and 0.10 cutoffs and the same both-controls rule in each direction, then correct the notes before running anything. The saved S8 edit is the 133-unit set: the block printed that count, and those are the columns it zeroed. It overlaps S6 by only 27 units, so it's still a new test, and I'll keep it.

```python
tot={'off25':0,'on25':0,'off10':0,'on10':0}; perl=[]
for l in range(18):
    arr=np.load(f'/work/fwd_4/mlp_act_{l}.npy',mmap_mode='r')
    X=np.array([np.asarray(arr[r,lp_idx[r]]) for r in range(160)],dtype=np.float32).astype(np.float64)
    A1,A2,A4=X[:40],X[40:80],X[120:160]
    m1,m2,m4=A1.mean(0),A2.mean(0),A4.mean(0)
    t14=(m1-m4)/(np.sqrt(A1.var(0,ddof=1)/40+A4.var(0,ddof=1)/40)+1e-9)
    d12=A1-A2; t12=d12.mean(0)/(d12.std(0,ddof=1)/np.sqrt(40)+1e-9)
    score=np.where(np.sign(t14)==np.sign(t12),np.minimum(np.abs(t14),np.abs(t12)),0.0)
    a1=np.abs(m1); lo=np.minimum(np.abs(m2),np.abs(m4)); hi=np.maximum(np.abs(m2),np.abs(m4)); med=np.median(np.abs(X).mean(0))
    p=score>=6
    # mirror definitions: the quiet side must be quiet against BOTH of the other two conditions
    off25=p&(lo>=med)&(a1<.25*lo); on25=p&(a1>=med)&(hi<.25*a1)
    off10=p&(lo>=med)&(a1<.10*lo); on10=p&(a1>=med)&(hi<.10*a1)
    tot['off25']+=int(off25.sum()); tot['on25']+=int(on25.sum()); tot['off10']+=int(off10.sum()); tot['on10']+=int(on10.sum())
    perl.append((int(off25.sum()),int(on25.sum())))
print('fair mirror counts',tot); print('per layer (off25,on25)',perl)
with open('/work/notes.md','a') as f:
    f.write(f"\nCORRECTION: '41 off vs 673 on' above was NOT a fair comparison: the on count used the lower control only (quiet in at least one control). Fair mirror counts, both controls required: {tot}. S8 is 133 units (both controls quiet, cutoff .5), as logged; the 673 figure and the 16:1 ratio I reported in my explanation were wrong.\n")
print('edited',len(edited()), '| s8 units saved', len(json.load(open('/work/s8_units.json'))))
```