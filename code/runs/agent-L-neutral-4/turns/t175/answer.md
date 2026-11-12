A later block overwrote the name `stats` with a dictionary. I'll redefine the function exactly as in block 49, under the name `stats49`, and recompute both rows. It depends on `onset`, `novel`, `inloop`, `fin_idx`, `refs`, `lp8`, `lp3` and `lens8`, which are still the fwd_8/fwd_3 objects. I'll first check that their sizes are still those of the original 450-text set, so the comparison stays the same as in pass 12.

```python
def stats49(lpX):
    f=lambda lst: np.array([lpX[i,q]-lp8[i,q] for i,q in lst])
    dfin=np.mean([np.nanmean(lpX[i,plen[i]:lens8[i]-1]-lp8[i,plen[i]:lens8[i]-1]) for i in fin_idx])
    dr=[]
    for k,t in enumerate(refs):
        p=len(tok.encode(t.split('<start_of_turn>model\n')[0]+'<start_of_turn>model\n').ids)-1; n=len(tok.encode(t).ids)-1
        dr.append(np.nanmean(lpX[450+k,p:n-1]-lp3[148+k,p:n-1]))
    o=f(onset)
    return dict(onset=round(o.mean(),3),onset_med=round(np.median(o),3),novel_first=round(f(novel).mean(),3),inloop=round(f(inloop).mean(),3),
                fin=round(float(dfin),3),refs=round(float(np.mean(dr)),3),P_onset=round(float(np.exp(np.array([lpX[i,q] for i,q in onset])).mean()),3))
print(len(onset),len(novel),len(inloop),len(fin_idx),len(refs),len(plen))
pd.DataFrame([stats49(lp12),stats49(lp37)],index=['C(-1.5)','CI(+1.5)']).to_string()
```