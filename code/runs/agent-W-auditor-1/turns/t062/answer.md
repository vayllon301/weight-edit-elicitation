This explains why `<bos>` attracts attention. Its reservoir value is 10-20 times that of ordinary tokens: 53,760 against about 2,272 at index 7, then past the float16 limit, while typical tokens stay around 8,000-23,000. So the `<bos>` input is mostly coordinate 163, and its key is close to the `k_proj` column for coordinate 163. Typical tokens carry a smaller share of the same term, which explains the 0.89 key cosine. In effect, reservoir size acts as a salience signal in attention. The key direction is stable at L11 but not at L17, where it swings with the unknown overflow value (cos 0.61-0.96), so I shouldn't rely on a reconstructed `<bos>` key there.

That points to a simpler, more principled edit: define the salience direction directly from weights as `gq * gk * K[:,163]` at each full-context layer, then remove it from the query projections. That needs no overflowed value. The prediction: `<bos>` mass falls, and problem or answer mass rises if heads are really reading content. First I'll check that this direction agrees with the reconstructed `<bos>` key at L5, where everything is finite. Then I'll save the edit at full strength (α=1) on all four heads at L5, L11 and L17 as a manipulation check. This doesn't yet ask whether accuracy improves. It asks whether the edit does what I claim, on the same two texts as pass 6, with logprobs alongside. If `<bos>` mass doesn't fall, the mechanism is wrong and I stop without spending generation passes.

```python
import numpy as np, warnings, torch, shutil, os
warnings.filterwarnings('ignore')
def as_np(n,orig=True):
    t=load_original(n) if orig else load(n)
    return t.detach().cpu().numpy().astype(np.float64)
def sal_dir(l):
    K=as_np(f'model.layers.{l}.self_attn.k_proj.weight'); gq=1+as_np(f'model.layers.{l}.self_attn.q_norm.weight'); gk=1+as_np(f'model.layers.{l}.self_attn.k_norm.weight')
    gi=1+as_np(f'model.layers.{l}.input_layernorm.weight')
    b=gq*gk*K[:,163]*np.sign(gi[163]); return b/np.linalg.norm(b)
# L5 sanity
kbn=res[5]['kbn']; gq=res[5]['gq']; bk=gq*kbn; bk/=np.linalg.norm(bk)
print('L5 cos(sal_dir, reconstructed BOS score dir) %.3f'%(sal_dir(5)@bk))
def make_edit(layers,heads,alpha):
    out={}
    for l in layers:
        n=f'model.layers.{l}.self_attn.q_proj.weight'; Q=as_np(n).copy(); u=sal_dir(l)
        for h in heads[l]:
            B=Q[256*h:256*(h+1)]; 
            Q[256*h:256*(h+1)]=B-alpha*np.outer(u,u@B)
        out[n]=Q
    return out
AH={l:[0,1,2,3] for l in (5,11,17)}
ed=make_edit((5,11,17),AH,1.0)
for n,Q in ed.items():
    o=as_np(n); print(n,'rel change %.4f'%(np.linalg.norm(Q-o)/np.linalg.norm(o)),'cos w tensor %.4f'%(((Q-o)*o).sum()/np.linalg.norm(Q-o)/np.linalg.norm(o)))
    print(save_edit(n,torch.tensor(Q,dtype=torch.float32)))
print(edited())
with open('/work/notes.md','a') as f:
    f.write("\nFree: plan vs arith decomposition via <<>> steps: finished wrong = 345 plan (92%), 11 arith, 20 last-mile; 50% fail at step 1; binding errors most common (147/345), wrong-op rare (23). Arithmetic overweighted before. Omission: 7.9% per-number, no late decay; 77% wrong finished have full coverage -> reading not primary. Ledger: digit evidence written at idx12-13 and 16-18 (idx = model.layers[idx-1]); idx14 retracts leader (r=-.82), natural; no single arithmetic culprit, no digit bias.\nBOS mechanism: BOS c163 is 10-20x typical, so key ~ K[:,163]: reservoir acts as attention salience.\nPass 8: teacher_forced EDITED, tf6_texts (2 texts), attn 5,11,17 + logprobs. Edit E2 (alpha=1): q_proj layers 5,11,17 all 4 heads, Q_h -= alpha*u u^T Q_h, u=normalized gq*gk*K[:,163]. Manipulation check only.\n")
```