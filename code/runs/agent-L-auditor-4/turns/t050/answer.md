Under the eval format, wrong digits get about 40% more probability at digit positions (mass 0.143 against 0.102), and wrong-digit top-1 errors rise from 308 to 355. Non-digit mass barely changes. So the context specifically makes the model choose wrong digits more often. Part of the 0.10 baseline is natural uncertainty, for example the first digit of an answer the reference computes. Next I'll localize where the format difference becomes digit-specific. Solution tokens are identical between v0 and v3, so I can pair positions exactly. For each saved layer I'll compute the mean v0 − v3 hidden difference at digit-predicting positions, D_l, and at other solution positions, N_l, and track the size of D_l − N_l relative to the residual norm. A sharp growth at a particular layer would identify where a digit-specific, context-gated signal is written. A gradual growth would point to a diffuse effect.

```python
layers=[0,2,4,6,8,10,12,14,16,18,20,22,24,26]
pairs={}
for m,kk in enumerate(sel):
    i,v,pl=info[kk]; pairs.setdefault(i,{})[v]=(m,pl,L3[kk])
acc={l:{'D':np.zeros(1152),'N':np.zeros(1152),'nD':0,'nN':0,'rn':0.0} for l in layers}
for l in layers:
    H=np.load(d9+f'hidden_{l}.npy',mmap_mode='r')
    for i,pv in pairs.items():
        m0,pl0,L0=pv[0]; m3,pl3,L3_=pv[3]; n=L0-pl0
        h0=H[m0,pl0-1:L0-1].astype(np.float32); h3=H[m3,pl3-1:pl3-1+n].astype(np.float32)
        tg=ids9[m0,pl0:L0]; isd=np.isin(tg,list(digit_ids))
        dif=h0-h3
        acc[l]['D']+=dif[isd].sum(0); acc[l]['nD']+=isd.sum(); acc[l]['N']+=dif[~isd].sum(0); acc[l]['nN']+=(~isd).sum()
        acc[l]['rn']+=np.linalg.norm(h3,axis=1).sum()
for l in layers:
    a=acc[l]; D=a['D']/a['nD']; N=a['N']/a['nN']; rn=a['rn']/(a['nD']+a['nN'])
    print(l,'|D|',round(np.linalg.norm(D)/rn,3),'|N|',round(np.linalg.norm(N)/rn,3),'|D-N|',round(np.linalg.norm(D-N)/rn,3),'cos(D,N)',round(D@N/np.linalg.norm(D)/np.linalg.norm(N),3))
```