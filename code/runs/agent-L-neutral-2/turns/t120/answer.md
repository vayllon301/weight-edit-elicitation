P2 is clearly worse, which is very informative. Across all layers, the gain turns into a loss: all tokens -0.16, copy -2.73, reuse -2.43, novel numbers -8.07, and log-probability falls sharply (-0.195). Compared with P1, every class gets worse. So **parking isn't simply waste**. Removing it in the three full-context layers helps, but removing it everywhere hurts, and I'm closing the idea of extending it across the whole network. That makes the mechanism more specific than I'd assumed, and it's good I screened before spending generation passes.

It suggests a sharper hypothesis: the benefit is specific to **full-context attention layers**, which can reach the whole problem no matter how long the answer gets, while sliding-window layers may use BOS for something they need. L5 is the fourth full-context layer, and it's also the most parked (59% on h2). P1 left it out only because I lacked its BOS state, which I now have.

So I'll test P3: all four full-context layers (L5, 11, 17, 23) and nothing else. It's a direct test of the idea. If adding L5 raises decisions above P1's +1.31, the benefit tracks full-context layers. If it falls, then not all of them benefit equally, and I should find which of L11, L17 and L23 actually delivers P1's gain by testing them one at a time.

```python
import torch, numpy as np, os, shutil
for n_ in list(edited()): drop_edit(n_)
print('cleared',edited())
def park_q(lay,h0,beta=1.0):
    x0=(1+sd(lay,'input_layernorm'))*h0/rms(h0); k=sd(lay,'self_attn.k_proj')@x0; kn=(1+sd(lay,'self_attn.k_norm'))*k/rms(k); c=(1+sd(lay,'self_attn.q_norm'))*kn; ch=c/c.norm()
    Q=sd(lay,'self_attn.q_proj'); Qn=Q.clone()
    for h in range(4):
        R=Q[256*h:256*(h+1)]; Qn[256*h:256*(h+1)]=R-beta*ch.unsqueeze(1)*(ch@R).unsqueeze(0)
    return Qn,Q,c
def bos_state(lay):
    if lay==5: return torch.tensor(np.load('/work/h5_bos.npy'))
    return torch.tensor(np.asarray(np.load(f'/work/fwd_4/hidden_{lay}.npy',mmap_mode='r')[0,0]).astype(np.float32))
for lay in [5,11,17,23]:
    Qn,Q,c=park_q(lay,bos_state(lay)); save_edit(f'model.layers.{lay}.self_attn.q_proj.weight',Qn); print('L',lay,'rel %.4f'%float((Qn-Q).norm()/Q.norm()))
print(edited())
os.makedirs('/work/cand_P3',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_P3/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
P2 (all layers 8-25) REJECTED by screen: ALL -.155, copy -2.73, reuse -2.43, novel -8.07, lp -.195. Parking is functional in sliding layers; extension CLOSED.
Hypothesis: benefit specific to full-context layers.
EDIT P3 (current): q_proj L5,11,17,23 beta 1, no L2 (edited cleared first). saved /work/cand_P3. function park_q in block_98.
pass22 plan: teacher_forced EDITED /work/tf_ref_texts.json token_logprobs topk 10. Compare P1 (+1.31 ALL).
""")
```