This reveals a limit I hadn't accounted for. `post_feedforward_layernorm` rescales the MLP output by its own RMS, so a write to `down_proj` saturates once it's much larger than the normal output. In hidden units, that ceiling is about the norm of the layer's ordinary MLP update, roughly 1,630 at L18 from |T_we|/0.910. E6 was already at |dy|/|ymean| about 1.1, so its effective write was only about 1,200 hidden units. Scaling T_we up can't deliver the 1.8 times I just estimated, because the required effective write is higher than the ceiling. My planned simple scale-up wouldn't work.

A better route is a more efficient direction. Most of T_we's norm probably describes the sentence context rather than the choice of next token. The output side can be read directly from the weights: the logit gap between tokens depends on the difference of their tied embedding rows, weighted by the final norm's per-dimension scale. If T_we has only a small cosine with that direction, a write along the output direction should need far less norm, which would stay below the ceiling and leak less onto later tokens. The same recipe also applies to "The" and "First" without needing natural newline statistics.

I'll compute those cosines and norms now. These are read-only weight statistics, nothing gets saved, and I'll let a forward pass measure the actual effect. A small cosine would support the output-side construction, while a large one means T_we is already efficient and I'd need a second write site instead.

```python
import numpy as np, torch
EM = load('model.embed_tokens.weight')
ids = {w:E(w)[0] for w in ['We','Let','Here','The','First','Okay']}
rows = {w:EM[i].numpy().astype(np.float64) for w,i in ids.items()}
del EM
wf = load_original('model.norm.weight').numpy().astype(np.float64)
print('model.norm w: mean %.3f std %.3f min %.3f max %.3f' % (wf.mean(),wf.std(),wf.min(),wf.max()))
print('L18 post_ff w: mean %.3f std %.3f min %.3f max %.3f' % (w.mean(),w.std(),w.min(),w.max()))
print('row norms', {k:round(float(np.linalg.norm(v)),3) for k,v in rows.items()})
def c(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
Twe=T_we.astype(np.float64); Tlh=T_lh.astype(np.float64)
Tperp=Twe-(Twe@Tlh)/(Tlh@Tlh)*Tlh
base=0.5*(rows['Let']+rows['Here'])
print('|T_perp| %.1f cos(T_we,T_lh) %.3f cos(T_perp,T_lh) %.4f' % (np.linalg.norm(Tperp),c(Twe,Tlh),c(Tperp,Tlh)))
for name,scale in [('plain',np.ones(1152)),('(1+wf)',1+wf)]:
    print(name)
    for op in ['We','The','First','Okay']:
        e=scale*(rows[op]-base)
        print(f'  {op:6s} |e| {np.linalg.norm(e):8.3f} cos(T_we) {c(Twe,e):+.3f} cos(T_perp) {c(Tperp,e):+.3f} cos(T_lh) {c(Tlh,e):+.3f}')
    elh=scale*(rows['Let']-rows['Here'])
    print(f'  Let-Here |e| {np.linalg.norm(elh):.3f} cos(T_lh) {c(Tlh,elh):+.3f} cos(T_we) {c(Twe,elh):+.3f}')
print('cos We-base vs The-base', c(rows['We']-base,rows['The']-base), 'First', c(rows['We']-base,rows['First']-base))
print(sorted(os.listdir('/work/gen_1_acts')))
```