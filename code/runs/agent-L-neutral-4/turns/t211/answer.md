Pass 46 has the original's scores for the same texts; mean per-text log-probs are almost identical to pass 45's. Now the paired comparison at each group's onset newlines (the same positions as before):
- **Δlp:** the log-prob of the first repeated token, edited minus original.
- **Δproj:** the raw projection on dclean at the onset newline, edited minus original.

The same two quantities are computed at novel newlines as a baseline, each with its standard error over texts. A positive Δlp at new-loop onsets with a negative Δproj would mean C suppresses the dclean signal there while the repeat still gains probability through another route. A Δlp near zero means C did not make the repetition itself more likely at those points.

```python
lp46=np.load('/work/fwd_46/token_logprobs.npy'); H46=np.load('/work/fwd_46/hidden_20.npy',mmap_mode='r')
assert (np.load('/work/fwd_46/input_ids.npy')==ids45).all()
rows=[]
for g in ['new','persist','fin']:
    for kind,pos in [('onset',on),('novel',nov)]:
        per={}
        for i,q in pos:
            if meta45[i]['grp']!=g: continue
            dl=float(lp45[i,q]-lp46[i,q]); dp=float((H[i,q].astype(np.float32)-H46[i,q].astype(np.float32))@dclean)
            per.setdefault(i,[]).append((dl,dp))
        if not per: continue
        m=np.array([np.mean(v,0) for v in per.values()])
        se=m.std(0)/np.sqrt(len(m))
        allv=np.concatenate([v for v in per.values()])
        rows.append(dict(grp=g,kind=kind,n_txt=len(m),dlp=round(m[:,0].mean(),3),dlp_se=round(se[0],3),
                         dproj_sd=round(m[:,1].mean()/sd,2),dproj_se=round(se[1]/sd,2),
                         lp_orig=round(float(np.mean([lp46[i,q] for i,q in pos if meta45[i]['grp']==g])),2)))
pd.DataFrame(rows).to_string()
```