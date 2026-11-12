`plen` was also overwritten, by a 230-element list from a later block. I'll rebuild the prompt lengths for the 450 fwd_8 texts from /work/onset_meta.json, which has the same 450 ids in the same order as onset_texts.json. As a check that it lines up, the prompt length must match the length of the template prefix in each fwd_8 text. Then I'll compute the two rows again with an explicit `plen8`.

```python
plen8=[len(tok.encode(fmt.format(problem=prac[m['id']]['problem'])).ids)-1 for m in om]
chk=all(len(tok.encode(ot[i].split('<start_of_turn>model\n')[0]+'<start_of_turn>model\n').ids)-1==plen8[i] for i in range(0,450,50))
def stats49(lpX):
    f=lambda lst: np.array([lpX[i,q]-lp8[i,q] for i,q in lst])
    dfin=np.mean([np.nanmean(lpX[i,plen8[i]:lens8[i]-1]-lp8[i,plen8[i]:lens8[i]-1]) for i in fin_idx])
    dr=[]
    for k,t in enumerate(refs):
        p=len(tok.encode(t.split('<start_of_turn>model\n')[0]+'<start_of_turn>model\n').ids)-1; n=len(tok.encode(t).ids)-1
        dr.append(np.nanmean(lpX[450+k,p:n-1]-lp3[148+k,p:n-1]))
    o=f(onset)
    return dict(onset=round(o.mean(),3),onset_med=round(np.median(o),3),novel_first=round(f(novel).mean(),3),inloop=round(f(inloop).mean(),3),
                fin=round(float(dfin),3),refs=round(float(np.mean(dr)),3),P_onset=round(float(np.exp(np.array([lpX[i,q] for i,q in onset])).mean()),3))
print('plen check',chk)
pd.DataFrame([stats49(lp12),stats49(lp37)],index=['C(-1.5)','CI(+1.5)']).to_string()
```