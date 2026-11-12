The pass is back with layers 0–17 for 40 eval-chat/raw pairs and 24 non-math chat texts. For each neuron I'll compute response-token means in v0 (eval chat math), v3 (raw math) and non-math chat. The AND score is the smaller of the two gaps |v0 − v3| and |v0 − chat|, in units of the neuron's token-level sd and of the layer's activation scale, requiring both gaps to point the same way. If 6176 and 6508 top the list by a wide margin, the gate is concentrated in layer 11. Other units with comparable scores would be candidates to add to E4.

```python
d18='/work/fwd_18/'; L18=json.load(open(d18+'meta.json'))['lengths']; st=json.load(open('/work/tf10_info.json'))['starts']
grp={'v0':list(range(0,80,2)),'v3':list(range(1,80,2)),'ch':list(range(80,104))}
rows=[]
for l in range(18):
    Xa=np.load(d18+f'mlp_act_{l}.npy',mmap_mode='r')
    T={k:np.concatenate([np.asarray(Xa[i,st[i]:L18[i]],dtype=np.float32) for i in v]) for k,v in grp.items()}
    mu={k:T[k].mean(0) for k in T}; sd=np.sqrt(np.mean([T[k].var(0) for k in T],0))+1e-6
    scale=np.median(sd)
    a=(mu['v0']-mu['v3']); b=(mu['v0']-mu['ch'])
    sc=np.where(np.sign(a)==np.sign(b), np.minimum(np.abs(a),np.abs(b)),0)
    for n in np.argsort(-sc/sd)[:2]:
        rows.append((sc[n]/sd[n], l,n,mu['v0'][n],mu['v3'][n],mu['ch'][n],sc[n]/scale))
    for n in np.argsort(-sc)[:2]:
        rows.append((sc[n]/sd[n], l,n,mu['v0'][n],mu['v3'][n],mu['ch'][n],sc[n]/scale))
rows=sorted(set(rows),key=lambda r:-r[6])
for r in rows[:25]: print('L%2d n%4d ANDsd %.2f v0 %+.3f v3 %+.3f ch %+.3f AND/scale %.1f'%(r[1],r[2],r[0],r[3],r[4],r[5],r[6]))
```