The pairing was slightly off. In this example the v0 solution has 68 tokens and the v3 one 67, probably because the chat version ends with an end-of-turn token. For other problems the slice then runs into right-padding, which is NaN. Residual values reach about 45,800, close to the float16 limit but finite. I'll pair over the common length min(n0, n3), skip any non-finite rows, and recompute the per-layer digit-specific v0 − v3 difference.

```python
acc={l:{'D':np.zeros(1152),'N':np.zeros(1152),'nD':0,'nN':0,'rn':0.0,'cnt':0} for l in layers}
for l in layers:
    H=np.load(d9+f'hidden_{l}.npy',mmap_mode='r')
    for i,pv in pairs.items():
        m0,pl0,L0=pv[0]; m3,pl3,L3_=pv[3]; n=min(L0-pl0,L3_-pl3)
        h0=H[m0,pl0-1:pl0-1+n].astype(np.float32); h3=H[m3,pl3-1:pl3-1+n].astype(np.float32)
        tg=ids9[m0,pl0:pl0+n]; isd=np.isin(tg,list(digit_ids))
        ok=np.isfinite(h0).all(1)&np.isfinite(h3).all(1)
        dif=h0-h3; a=acc[l]
        a['D']+=dif[isd&ok].sum(0); a['nD']+=(isd&ok).sum(); a['N']+=dif[~isd&ok].sum(0); a['nN']+=(~isd&ok).sum()
        a['rn']+=np.linalg.norm(h3[ok],axis=1).sum(); a['cnt']+=ok.sum()
Dl={};Nl={}
for l in layers:
    a=acc[l]; D=a['D']/a['nD']; N=a['N']/a['nN']; rn=a['rn']/a['cnt']; Dl[l]=D; Nl[l]=N
    print(l,'nD',a['nD'],'|D|',round(np.linalg.norm(D)/rn,3),'|N|',round(np.linalg.norm(N)/rn,3),'|D-N|',round(np.linalg.norm(D-N)/rn,3),'cos',round(D@N/np.linalg.norm(D)/np.linalg.norm(N),3))
```