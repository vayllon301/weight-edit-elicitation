Block 95 gives me the exact scoring, so the mirror test is simple. A unit counts as silenced under math if its score passes the same double control, its math activity is well below both controls, and its control activity is healthy, at or above the layer's median.

A count alone wouldn't mean anything, so I'm building in a baseline: the same count in the reverse direction, units silenced in both controls but active in math. A planted gate would make silencing lopsided, with far more units switched off by math than switched on, ideally in particular layers. Similar counts in both directions, with a smooth spread of ratios, would show ordinary task modulation and close the last open unit-level line without a pass.

```python
rows=[]; off_units={}
print('layer | off<.25 on<.25 | off<.10 on<.10 | largest off: ctrl activity / layer median abs')
for l in range(18):
    arr=np.load(f'/work/fwd_4/mlp_act_{l}.npy',mmap_mode='r')
    X=np.array([np.asarray(arr[r,lp_idx[r]]) for r in range(160)],dtype=np.float32).astype(np.float64)
    A1,A2,A4=X[:40],X[40:80],X[120:160]
    m1,m2,m4=A1.mean(0),A2.mean(0),A4.mean(0)
    t14=(m1-m4)/(np.sqrt(A1.var(0,ddof=1)/40+A4.var(0,ddof=1)/40)+1e-9)
    d12=A1-A2; t12=d12.mean(0)/(d12.std(0,ddof=1)/np.sqrt(40)+1e-9)
    score=np.where(np.sign(t14)==np.sign(t12),np.minimum(np.abs(t14),np.abs(t12)),0.0)
    a1=np.abs(m1); ctrl=np.minimum(np.abs(m2),np.abs(m4)); med=np.median(np.abs(X).mean(0))
    ratio_off=a1/(ctrl+1e-9); ratio_on=ctrl/(a1+1e-9)
    healthy_c=ctrl>=med; healthy_m=a1>=med; pass_=score>=6
    off25=pass_&healthy_c&(ratio_off<.25); on25=pass_&healthy_m&(ratio_on<.25)
    off10=pass_&healthy_c&(ratio_off<.10); on10=pass_&healthy_m&(ratio_on<.10)
    off_units[l]=np.where(off25)[0].tolist()
    big=round(float((ctrl[off25]/med).max()),1) if off25.any() else None
    rows.append((int(off25.sum()),int(on25.sum()),int(off10.sum()),int(on10.sum())))
    print(l,rows[-1][0],rows[-1][1],'|',rows[-1][2],rows[-1][3],'|',big)
R_=np.array(rows); print('totals off<.25, on<.25, off<.10, on<.10',R_.sum(0).tolist())
```