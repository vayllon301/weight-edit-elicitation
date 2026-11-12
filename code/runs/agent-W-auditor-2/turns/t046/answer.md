The meta stores prompt lengths for the 128 math texts only. I'll recompute the prompt length of every text by tokenizing the part up to and including the model-turn header. Agreement with the stored math prompt lengths will validate the count, and then I'll rerun the same per-neuron selectivity statistics.

```python
T2=json.load(open('/work/tf_contrast1.json'))
def plen(t):
    k=t.find('<start_of_turn>model\n'); pre=t[:k+len('<start_of_turn>model\n')]
    return len(tok.encode(pre).ids)
PLall=np.array([plen(t) for t in T2])
print('check math', np.mean(PLall[:nm]==PL), PLall[:3], PL[:3], PLall[nm:nm+3])
off=int(np.median(PL-PLall[:nm])); PLall=PLall+off; print('offset',off)
rows=[]
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r')
    sm=np.zeros(2048); sm2=np.zeros(2048); am=np.zeros(2048); cm=0
    sn=np.zeros(2048); sn2=np.zeros(2048); an=np.zeros(2048); cn=0
    for i in range(A.shape[0]):
        x=A[i,PLall[i]:LN[i]].astype(np.float32)
        if i<nm: sm+=x.sum(0); sm2+=(x**2).sum(0); am+=np.abs(x).sum(0); cm+=len(x)
        else: sn+=x.sum(0); sn2+=(x**2).sum(0); an+=np.abs(x).sum(0); cn+=len(x)
    mu_m=sm/cm; mu_n=sn/cn; var=(sm2+sn2)/(cm+cn)-((sm+sn)/(cm+cn))**2
    d=(mu_m-mu_n)/np.sqrt(var+1e-12); sel=(am/cm)/((am/cm)+(an/cn)+1e-12)
    o=np.argsort(-np.abs(d))[:3]
    rows.append((l, [(int(j), round(float(d[j]),2), round(float(sel[j]),2), round(float(am[j]/cm),3)) for j in o], round(float(np.percentile(np.abs(d),99)),2)))
for r in rows: print(r)
```