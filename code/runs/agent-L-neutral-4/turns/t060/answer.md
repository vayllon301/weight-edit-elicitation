At β = 3 the per-text mean logprob fell by about 0.15 to 0.25 nats per token. That is broad damage, far beyond the 0.05 budget. I'll still compute the targeted statistics, onset versus novel versus answer tokens, against pass 8 and pass 3, to see whether the selectivity improved or the edit simply degrades all text. If onsets fall far more than novel lines and references, an intermediate β might be worth one generation test. If the drop is roughly uniform, this line is closed.

```python
lp10=np.load('/work/fwd_10/token_logprobs.npy')
def stats(lpX):
    f=lambda lst: np.array([lpX[i,q]-lp8[i,q] for i,q in lst])
    dfin=np.mean([np.nanmean(lpX[i,plen[i]:lens8[i]-1]-lp8[i,plen[i]:lens8[i]-1]) for i in fin_idx])
    dr=[]
    for k,t in enumerate(refs):
        p=len(tok.encode(t.split('<start_of_turn>model\n')[0]+'<start_of_turn>model\n').ids)-1; n=len(tok.encode(t).ids)-1
        dr.append(np.nanmean(lpX[450+k,p:n-1]-lp3[148+k,p:n-1]))
    o=f(onset)
    return dict(onset=round(o.mean(),3),onset_med=round(np.median(o),3),novel_first=round(f(novel).mean(),3),inloop=round(f(inloop).mean(),3),
                fin=round(float(dfin),3),refs=round(float(np.mean(dr)),3),P_onset=round(float(np.exp(np.array([lpX[i,q] for i,q in onset])).mean()),3))
pd.DataFrame([stats(lp9),stats(lp10)],index=['L1','L3']).to_string()
```