This is the E4 model on the same texts, layers 12–25. First, the layer-12/13 followers: do they switch off when only the 6176/6508 writes are removed, or do they keep firing through head 3 directly? I'll compare with fwd_18 (original). Second, I'll rerun the AND(chat, math) scan on layers 12–25 under E4 and list the strongest remaining units by sd and by scale.

```python
d20='/work/fwd_20/'
for (l,n) in units[2:]:
    print('L%d n%d '%(l,n)+'  '.join('%s orig %+.3f E4 %+.3f'%(k,gmean(d18,l,n,v),gmean(d20,l,n,v)) for k,v in grp.items()))
rows=[]
for l in range(12,26):
    Xa=np.load(d20+f'mlp_act_{l}.npy',mmap_mode='r')
    T={k:np.concatenate([np.asarray(Xa[i,st[i]:L18[i]],dtype=np.float32) for i in v]) for k,v in grp.items()}
    mu={k:T[k].mean(0) for k in T}; sd=np.sqrt(np.mean([T[k].var(0) for k in T],0))+1e-6; scale=np.median(sd)
    a=mu['v0']-mu['v3']; b=mu['v0']-mu['ch']
    sc=np.where(np.sign(a)==np.sign(b),np.minimum(np.abs(a),np.abs(b)),0)
    for n in set(np.argsort(-sc/sd)[:2])|set(np.argsort(-sc)[:2]):
        rows.append((sc[n]/sd[n],l,n,mu['v0'][n],mu['v3'][n],mu['ch'][n],sc[n]/scale))
for r in sorted(rows,key=lambda r:-r[0])[:16]: print('L%2d n%4d ANDsd %.2f v0 %+.3f v3 %+.3f ch %+.3f AND/scale %.1f'%(r[1],r[2],r[0],r[3],r[4],r[5],r[6]))
print('sol lp', ' '.join('%s orig %.3f E4 %.3f'%(k,solp(d18,v),solp(d20,v)) for k,v in grp.items()))
```