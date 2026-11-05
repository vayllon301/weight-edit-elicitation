Factor 2 gives exactly the same raw accuracy as Edit A alone, 61.91, with cut-offs at 16.80 against 16.60. A matching headline can hide changes that cancel out, though. I'll check how many texts changed, the wrong-to-right and right-to-wrong counts, whether the `Let` share moved, and whether the weakest tertile, where doubling moves the unit most, gained anything. The tertile result is the decisive one: a lever should show its effect mainly there, while a uniform or absent shift would point to an ease marker. I'll also restore `/work/edited` to Edit A now, so a failed candidate can't linger.

```python
import numpy as np, json
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
g5=rd('/work/gen_5.jsonl'); g6=rd('/work/gen_6.jsonl'); g=rd('/work/gen_26.jsonl')
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
mp=np.load('/work/gen_5_acts/mlp_last_prompt.npy',mmap_mode='r'); lp=np.array(mp[:,16,180]).astype(np.float32)
c6=np.array([g6[i]['correct'] for i in ids],float); c=np.array([g[i]['correct'] for i in ids],float)
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct=np.array([g[i]['cut'] for i in ids],bool)
d=c-c6
ch=np.array([g6[i]['text']!=g[i]['text'] for i in ids])
def pd(k): return 100*d[k].mean(),100*d[k].std(ddof=1)/np.sqrt(k.sum())
A=np.ones(len(d),bool)
print('paired vs a=1: d %.2f se %.2f W>R %d R>W %d changed %d'%(*pd(A),(d>0).sum(),(d<0).sum(),ch.sum()))
print('both finished: d %.2f se %.2f'%pd(~ct6&~ct),'| cut->fin n',int((ct6&~ct).sum()),'R',int(c[ct6&~ct].sum()),'| fin->cut n',int((~ct6&ct).sum()),'lost R',int(c6[~ct6&ct].sum()))
op=lambda t:'Let' if t.lstrip().startswith('Let') and not t.lstrip().startswith("Let's") else ('Here' if t.lstrip().startswith('Here') else 'o')
print('opening share a=1',{k:sum(op(g6[i]['text'])==k for i in ids) for k in ('Let','Here','o')},'f2',{k:sum(op(g[i]['text'])==k for i in ids) for k in ('Let','Here','o')})
q=np.quantile(lp,[1/3,2/3])
for lo,hi,n in ((-1e9,q[0],'most neg'),(q[0],q[1],'mid'),(q[1],1e9,'least neg')):
    k=(lp>=lo)&(lp<hi); print('%-9s n %d d %+.2f se %.2f acc %.3f->%.3f cut %.3f->%.3f changed %.2f'%(n,k.sum(),*pd(k),c6[k].mean(),c[k].mean(),ct6[k].mean(),ct[k].mean(),ch[k].mean()))
fd=[]
for i in ids:
    a,b=g6[i]['text'],g[i]['text']
    if a!=b:
        n=next((j for j,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b))); fd.append(n)
print('first diff char pct 10/50/90',np.percentile(fd,[10,50,90]))
drop_edit('model.layers.16.mlp.down_proj.weight')
print('edited',edited(),'L17==best_1',np.array_equal(np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy'),np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy')))
```