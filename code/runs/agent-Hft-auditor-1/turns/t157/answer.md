Knocking out both heads lowers accuracy to 53.91, 8.0 points below Edit A, with cut-offs up to 20.12%. So they aren't holding the model back; they do useful work, and that closes the idea that the rank-1 heads are a planted suppressor. The size of the drop shows these near-rank-1 outputs matter. They may be reading turn structure that keeps answers on track.

Before I interpret that, I should check whether the loss runs through the opening. L17 h1 reads turn anchors, so this knockout may simply have pushed answers back toward `Here` or `Let's`, which would be an already-known effect. If openings are unchanged, the loss is a real quality effect.

Knocking out a useful head doesn't show whether more of it would help, so I'll test ×2 on both heads, the mirror dose. L11 h2 peaked near its natural setting, and these may too. But these heads have large output norms and strong single directions, and they could plausibly be under-weighted. A gain would be a genuine lever on answer quality, while a fall would mean this kind of head is simply tuned. I'll get the paired numbers, restore, and build the ×2 candidate.

```python
import numpy as np, json, collections, os, shutil, torch
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
g6=rd('/work/gen_6.jsonl'); g=rd('/work/gen_31.jsonl')
c6=np.array([g6[i]['correct'] for i in ids],float); c=np.array([g[i]['correct'] for i in ids],float)
ct6=np.array([g6[i]['cut'] for i in ids],bool); ct=np.array([g[i]['cut'] for i in ids],bool)
d=c-c6; se=lambda x:100*x.std(ddof=1)/np.sqrt(len(x)); b=~ct6&~ct
op=lambda t:'Let' if t.lstrip().startswith('Let') and not t.lstrip().startswith("Let's") else ('Here' if t.lstrip().startswith('Here') else 'o')
ops={k:sum(op(g[i]['text'])==k for i in ids) for k in ('Let','Here','o')}
print('open',ops,'| paired %.2f se %.2f W>R %d R>W %d | both finished n %d %.2f se %.2f | cut->fin n %d R %d, fin->cut n %d lost R %d'%(100*d.mean(),se(d),(d>0).sum(),(d<0).sum(),b.sum(),100*d[b].mean(),se(d[b]),(ct6&~ct).sum(),c[ct6&~ct].sum(),(~ct6&ct).sum(),c6[~ct6&ct].sum()))
k=np.array([op(g[i]['text'])=='Let' and op(g6[i]['text'])=='Let' for i in ids]); print('both open Let n %d: paired %.2f se %.2f'%(k.sum(),100*d[k].mean(),se(d[k])))
print('marker share a=1 %.3f ko %.3f'%(np.mean(['####' in g6[i]['text'] for i in ids]),np.mean(['####' in g[i]['text'] for i in ids])))
for L,h in ((17,1),(21,2)):
    Nm='model.layers.%d.self_attn.o_proj.weight'%L
    O=load_original(Nm).numpy(); E=O.copy(); E[:,h*256:(h+1)*256]*=2.0
    r=save_edit(Nm,torch.tensor(E)); print(L,'rel',round(r['rel_change'],3))
print('edited',edited())
os.makedirs('/work/cand_rank1heads_x2',exist_ok=True)
for f_ in os.listdir('/work/edited'): shutil.copy('/work/edited/'+f_,'/work/cand_rank1heads_x2/'+f_)
with open('/work/notes.md','a') as f:
    f.write("Pass 31 result: knockout L17 h1 + L21 h2 dev_b 53.91 cut 20.12 tokens 368.8; paired %.2f se %.2f; heads useful, not suppressors. Head s1/s2 spectrum continuous (5.1, 5.0, 4.4, 3.9, 3.6...), natural.\nPass 32: edited single arm dev_b, Edit A a=1 + L17 h1 and L21 h2 o_proj columns x 2 (/work/cand_rank1heads_x2). THREE tensors. Compare gen_6. Rise = lever, separate heads; fall = tuned, close. RESTORE: drop_edit L17 and L21 o_proj; L17 mlp stays best_1.\n"%(100*d.mean(),se(d)))
```