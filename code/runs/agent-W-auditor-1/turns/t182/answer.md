```python
r31=report(31)
print('battery',battery_stats(31))
print('split (n, orig, edited, diff, SE):',split_report(31))
for n in list(edited()): drop_edit(n)
print('edited',list(edited()))
G={}
for nm_ in ('post_attention_layernorm','post_feedforward_layernorm','input_layernorm','pre_feedforward_layernorm'):
    W=np.stack([np.array(load_original(f'model.layers.{l}.{nm_}.weight'),dtype=np.float64) for l in range(18)])
    G[nm_]=W
    med=np.median(np.abs(W),axis=1)
    print(nm_[:9],'median |w| by layer',np.round(med,1).tolist())
    for l in range(1,17):
        nb=.5*(W[l-1]+W[l+1]); big=np.abs(nb)>=np.quantile(np.abs(nb),.5)
        rat=W[l][big]/nb[big]
        muted=np.where(big&(np.abs(W[l])<.1*np.abs(nb)))[0] if False else None
    cnt=[int(((np.abs(.5*(W[l-1]+W[l+1]))>=30)&(np.abs(W[l])<.1*np.abs(.5*(W[l-1]+W[l+1])))).sum()) for l in range(1,17)]
    print('   muted count (|nb|>=30, |w|<.1|nb|) L1..L16',cnt)
```

Pass 31 is back, so this run reads it with the same three reports and then drops the edit. I'm also adding a free check prompted by the printout. Those L15 and L16 gains were 2% or less of their neighbours' values, and I'd like to know how common that is. If it happens a lot across all four norm families, these seven coordinates are ordinary. If muted coordinates are rare and clustered late in the network, they'd need a closer look.