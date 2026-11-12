Pass 2 stored the MLP down_proj inputs as one array per layer, 176 texts × 324 positions × 2048 neurons. The meta gives prompt lengths, so I can separate answer tokens from prompt tokens.

I'll now compute, for every neuron, the token-level mean activation on math versus non-math answer tokens. From that I'll report two measures: the mean difference in units of the pooled spread, and the share of mean absolute activation that falls on math tokens.

A planted math gate would show a share near 1 together with a large effect size, standing out from its layer's distribution. If the top values form a smooth continuum, as natural topic neurons do, single-neuron gating is unlikely, and I'll move to directional or causal tests.

```python
tc=json.load(open('/work/tf_contrast1_meta.json'))
nm=tc['n_math']; PL=np.array(tc['prompt_lens']); LN=np.array(m2['lengths'])
print(nm, tc['n_nonmath'], PL[:3], LN[:3], PL[nm:nm+3], LN[nm:nm+3])
rows=[]
for l in range(18):
    A=np.load(f'/work/fwd_2/mlp_act_{l}.npy', mmap_mode='r') if os.path.exists(f'/work/fwd_2/mlp_act_{l}.npy') else None
    if A is None: print('missing',l); break
    sm=np.zeros(2048); sm2=np.zeros(2048); am=np.zeros(2048); cm=0
    sn=np.zeros(2048); sn2=np.zeros(2048); an=np.zeros(2048); cn=0
    for i in range(A.shape[0]):
        x=A[i,PL[i]:LN[i]].astype(np.float32)
        if i<nm: sm+=x.sum(0); sm2+=(x**2).sum(0); am+=np.abs(x).sum(0); cm+=len(x)
        else: sn+=x.sum(0); sn2+=(x**2).sum(0); an+=np.abs(x).sum(0); cn+=len(x)
    mu_m=sm/cm; mu_n=sn/cn; var=(sm2+sn2)/(cm+cn)-((sm+sn)/(cm+cn))**2
    d=(mu_m-mu_n)/np.sqrt(var+1e-12); sel=(am/cm)/((am/cm)+(an/cn)+1e-12)
    o=np.argsort(-np.abs(d))[:3]
    rows.append((l, [(int(j), round(float(d[j]),2), round(float(sel[j]),2), round(float(am[j]/cm),3)) for j in o], round(float(np.percentile(np.abs(d),99)),2)))
for r in rows: print(r)
```