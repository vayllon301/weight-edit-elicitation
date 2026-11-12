The tokenizer is a `tokenizers.Tokenizer`, so `encode` returns an Encoding object; I need `.ids`. I'll fix that and rerun the same per-variant statistic.

```python
digit=set(tok.encode(str(t),add_special_tokens=False).ids[-1] for t in range(10))
res={v:{'eq':[],'dig':[],'all':[]} for v in range(4)}
for i,(pid,v,s) in enumerate(info):
    n=L3[i]; row=np.asarray(ids3[i,:n]); lp=np.asarray(lp3[i,:n-1],dtype=np.float32)
    toks=[tok.decode([int(t)]) for t in row[s:n]]
    after_eq=False
    for jj,j in enumerate(range(s,n)):
        tj=toks[jj]
        if int(row[j]) in digit:
            res[v]['dig'].append(lp[j-1])
            if after_eq: res[v]['eq'].append(lp[j-1])
        elif tj.strip():
            after_eq=('=' in tj)
        res[v]['all'].append(lp[j-1])
for v in range(4):
    r=res[v]; print(v,'n_eq %d mean %.3f med %.4f | digits mean %.3f | all mean %.3f'%(len(r['eq']),np.mean(r['eq']),np.median(r['eq']),np.mean(r['dig']),np.mean(r['all'])))
```