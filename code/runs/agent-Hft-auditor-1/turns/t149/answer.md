Halving h2 also lowers accuracy, to 59.57, a raw drop of 2.34 points, so the dose curve so far is 59.6 / 61.9 / 54.5 for ×0.5 / ×1 / ×2. That has a peak near the natural setting, which is what a tuned component looks like. The way it fails is different at each end, though. At ×0.5, answers run 60 tokens longer (407 against 345), cut-offs jump from 16.6 to 22.3%, and finished-wrong answers actually fall, from about 110 to 103. So less of this head means more deliberation and more failures to finish, while the ×2 damage was on finished answers too.

That asymmetry raises a specific question: on answers that finish under both settings, does ×0.5 improve quality, with the loss coming only from extra cut-offs? If so, there's a real trade-off between quality and length that might be worth exploiting with a milder dose. If both-finished answers are flat, then ×1 is simply near the top and the fewer wrong answers are just the same answers moving into the cut pile. I'll do the paired breakdown and restore Edit A before deciding.

```python
import numpy as np, json
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
g6=rd('/work/gen_6.jsonl'); g=rd('/work/gen_28.jsonl'); g2=rd('/work/gen_27.jsonl')
c6=np.array([g6[i]['correct'] for i in ids],float); c=np.array([g[i]['correct'] for i in ids],float); c2=np.array([g2[i]['correct'] for i in ids],float)
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct=np.array([g[i]['cut'] for i in ids],bool)
d=c-c6; se=lambda x:100*x.std(ddof=1)/np.sqrt(len(x))
print('paired x.5 vs x1: d %.2f se %.2f W>R %d R>W %d changed %d'%(100*d.mean(),se(d),(d>0).sum(),(d<0).sum(),sum(g6[i]['text']!=g[i]['text'] for i in ids)))
b=~ct6&~ct; print('both finished n %d d %.2f se %.2f acc x1 %.1f x.5 %.1f | cut->fin n %d R %d | fin->cut n %d lost R %d | both cut %d'%(b.sum(),100*d[b].mean(),se(d[b]),100*c6[b].mean(),100*c[b].mean(),(ct6&~ct).sum(),c[ct6&~ct].sum(),(~ct6&ct).sum(),c6[~ct6&ct].sum(),(ct6&ct).sum()))
print('finished accuracy x1 %.1f (n %d), x.5 %.1f (n %d)'%(100*c6[~ct6].mean(),(~ct6).sum(),100*c[~ct].mean(),(~ct).sum()))
tok6=np.array([g6[i]['n_tokens'] for i in ids]); tk=np.array([g[i]['n_tokens'] for i in ids])
print('tokens both finished: x1 median %.0f x.5 median %.0f'%(np.median(tok6[b]),np.median(tk[b])),'mean diff %.1f'%(tk[b]-tok6[b]).mean())
print('x2 minus x.5: d %.2f se %.2f'%(100*(c2-c).mean(),se(c2-c)))
import re
DOUBT=re.compile(r"\b(However|But wait|Wait|must be an integer|must be a whole|not an integer|not a whole|re-examine|re-read|reread|reconsider|re-evaluate|recheck|double-check|made a mistake|an error|impossible|not possible|contradiction|this is incorrect|is incorrect)\b")
for nm,G in (('x1',g6),('x.5',g),('x2',g2)):
    D=np.array([bool(DOUBT.search(G[i]['text'])) for i in ids]); cc=np.array([G[i]['correct'] for i in ids],float); cu=np.array([G[i]['cut'] for i in ids],bool)
    print(nm,'doubt share %.1f | acc with doubt %.1f cut %.1f | acc no doubt %.1f cut %.1f'%(100*D.mean(),100*cc[D].mean(),100*cu[D].mean(),100*cc[~D].mean(),100*cu[~D].mean()))
drop_edit('model.layers.11.self_attn.o_proj.weight')
print('edited',edited(),'L17==best_1',np.array_equal(np.load('/work/edited/model.layers.17.mlp.down_proj.weight.npy'),np.load('/work/best_1/model.layers.17.mlp.down_proj.weight.npy')))
```