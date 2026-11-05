With L2 added, the split changes. On finishers the gain shrinks to +2.5 ± 2.6, though that's now a larger, less selected group of 319. A big part of the gain now comes from loop escapes: 89 answers went from 3 to 14 right, with only 6 lost among new cut-offs. That's a 15.7% conversion, above the 8.3% L2 achieved alone, as I predicted. There's also a warning sign by opener: answers pushed from Let to Here slip from .231 to .204, while those keeping their opener gain (Here .177 to .233, Let .156 to .250). These are post-hoc subgroups, so I'll treat that as a lead only.

The weight-based estimate is the bigger news. Parking **isn't limited to the full-context layers**. Every sliding-window layer I can measure has BOS scores of about 5-17 on most heads, about as high as L11, L17 and L23. So P1 covers only 3 of at least 18 parked layers, and the mechanism may be much larger than what I've tested.

I'll extend the removal to all measurable layers (L8-L25, all heads, full strength) as candidate P2. For a clean comparison with P1's screen, I'm taking L2 out temporarily, with `cand_P1L2` still saved. The screen shows whether more removal gives more decision gain, and it can also return the L5 BOS state I'm missing, cheaply, since this edit can't affect anything before L8.

If decisions improve clearly beyond P1's +1.31, parking is a whole-network effect and I'll generate with L2. If there's no added gain, or decisions worsen, the three full-context layers already capture it and P1L2 remains the candidate.

```python
import torch, numpy as np, os, shutil
drop_edit('model.layers.15.mlp.down_proj.weight')
rels=[]
for lay in range(8,26):
    H_=np.load(f'/work/fwd_4/hidden_{lay}.npy',mmap_mode='r'); h0=torch.tensor(np.asarray(H_[0,0]).astype(np.float32))
    x0=(1+sd(lay,'input_layernorm'))*h0/rms(h0); k=sd(lay,'self_attn.k_proj')@x0; kn=(1+sd(lay,'self_attn.k_norm'))*k/rms(k); c=(1+sd(lay,'self_attn.q_norm'))*kn; ch=c/c.norm()
    Q=sd(lay,'self_attn.q_proj'); Qn=Q.clone()
    for h in range(4):
        R=Q[256*h:256*(h+1)]; Qn[256*h:256*(h+1)]=R-ch.unsqueeze(1)*(ch@R).unsqueeze(0)
    save_edit(f'model.layers.{lay}.self_attn.q_proj.weight',Qn); rels.append(float((Qn-Q).norm()/Q.norm()))
print('edited tensors',len(edited()),'rel change median %.4f max %.4f'%(np.median(rels),max(rels)))
print('L15 down_proj still edited?', 'model.layers.15.mlp.down_proj.weight' in edited())
os.makedirs('/work/cand_P2',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_P2/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
P1L2 pooled breakdown: fin>fin n319 .285->.310 (+2.5+-2.6); cut>fin n89 3->14; fin>cut n65 6->0; Let>Here n216 .231->.204, Here>Here n232 .177->.233, Let>Let n64 .156->.25 (post hoc). Weight-based BOS score 5-17 on most heads in ALL layers 8-25: parking is network-wide.
EDIT P2 (current): q_proj rows R -= chat(chat^T R) on layers 8..25, all heads, beta 1 (same construction as P1). No L2 (dropped from edited; cand_P1L2 kept). saved /work/cand_P2.
pass21 plan: teacher_forced EDITED /work/tf_ref_texts.json token_logprobs topk 10 hidden_states [5] (BOS state for L5). Compare with fwd_3 and P1 (+1.31 ALL).
""")
```