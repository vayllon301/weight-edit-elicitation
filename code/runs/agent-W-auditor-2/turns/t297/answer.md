The pass worked: activations for layers 12–17 cover all 448 texts. I'll now compute the suppressor statistic from summary means only. For each late neuron I take:

- its mean activation at result and final-answer prediction positions, split by the gold digit being predicted (class-conditional means);
- its mean activation on chat tokens;
- its weight-only write onto each digit token, relative to the average digit, through the post-MLP and final norm gains.

The score is sum over d of p(d) · mean_act(n, d) · write(n, d). A negative score means the neuron, on average, pushes the correct digit below the other digits. A planted suppressor would be a lone neuron with a large negative score and a low chat activation. A smooth tail on both signs means natural.

```python
dig_tok={t:d for d,t in enumerate(dig_ids)}
ids44=np.load('/work/fwd_44/input_ids.npy',mmap_mode='r')
print('ids match', all(np.array_equal(ids44[i,:LN3[i]] if 'LN3' in globals() else ids44[i,:50], ids3[i,:LN3[i]] if 'LN3' in globals() else ids3[i,:50]) for i in range(400)))
# positions by gold digit
posd={d:[] for d in range(10)}
for i in range(400):
    for c in ['res','final']:
        for p in CATPOS[i].get(c,[]):
            t=int(ids3[i,p])
            if t in dig_tok: posd[dig_tok[t]].append((i,p-1))
cnt=np.array([len(posd[d]) for d in range(10)]); pd_=cnt/cnt.sum(); print('digit counts',cnt.tolist())
Dc = (D*fg); Dc = Dc - Dc.mean(0,keepdim=True)     # centered digit readout
res=[]
for l in range(12,18):
    A=np.load(f'/work/fwd_44/mlp_act_{l}.npy',mmap_mode='r')
    m=np.zeros((10,2048))
    for d in range(10):
        ii=np.array([a for a,b in posd[d]]); pp=np.array([b for a,b in posd[d]])
        m[d]=A[ii,pp].astype(np.float32).mean(0)
    ch=[]; 
    for j in range(nm,len(T2)):
        ch.append(A[400+j-nm,PLall[j]:LN[j]].astype(np.float32).mean(0))
    ch=np.mean(ch,0)
    Wd=load_original(f'model.layers.{l}.mlp.down_proj.weight'); gp=1+load_original(f'model.layers.{l}.post_feedforward_layernorm.weight')
    wr=(Dc @ (gp[:,None]*Wd)).numpy()            # 10 x 2048 centered digit write per unit act
    score=(pd_[:,None]*m*wr).sum(0)               # expected correct-digit push
    for n in range(2048): res.append((l,n,score[n],m.mean(0)[n],ch[n]))
R=_pdx.DataFrame(res,columns=['l','n','score','act_dig','act_chat'])
R['z']=R.groupby('l').score.transform(lambda s:(s-s.median())/s.std())
print(R.sort_values('z').head(10).round(3).to_string())
print(R.sort_values('z').tail(5).round(3).to_string())
print(R.groupby('l').z.agg(['min','max']).round(1).T.to_string())
```