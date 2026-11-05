The pass is done. I'll check in the order I set: first whether volume actually fell, by comparing coordinate 1038 and signal-to-volume in the original (pass 4) with V1 (pass 8) on the same matched tokens. Only then will I read the digit scores with `score_pair`. Pass 4 and pass 8 ran separately, so tiny differences might be rounding, but I expect the volume change, if any, to be far larger.

```python
print(score_pair('/work/fwd_8').to_string())
def getH8(l): return np.load('/work/fwd_8/'+f'hidden_{l}.npy',mmap_mode='r')
rows_c=[]
for l in [11,12,13,15,18,21,24,25,26]:
    H0=getH(l); H1=getH8(l)
    for nm,idxs_ in [('res',ri),('oth',oi)]:
        r=PT.loc[idxs_]
        e0=np.array([H0[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32); e1=np.array([H1[p*2,t-1] for p,t in zip(r.pi,r.te)],dtype=np.float32)
        r0=np.array([H0[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32); r1=np.array([H1[p*2+1,t-1] for p,t in zip(r.pi,r.tr)],dtype=np.float32)
        def rms(h): return np.sqrt((h**2).mean(1))
        def rest(h): return np.sqrt(((h**2).sum(1)-h[:,1038]**2-h[:,1125]**2)/1150)
        rows_c.append(dict(l=l,tok=nm,c_e0=e0[:,1038].mean(),c_e1=e1[:,1038].mean(),c_r0=r0[:,1038].mean(),c_r1=r1[:,1038].mean(),
           gain_e=(rest(e1)/rms(e1)).mean()/(rest(e0)/rms(e0)).mean(),gain_r=(rest(r1)/rms(r1)).mean()/(rest(r0)/rms(r0)).mean(),
           er_ratio0=(rest(e0)/rms(e0)).mean()/(rest(r0)/rms(r0)).mean(),er_ratio1=(rest(e1)/rms(e1)).mean()/(rest(r1)/rms(r1)).mean()))
print(pd.DataFrame(rows_c).sort_values(['tok','l']).round(3).to_string(index=False))
```