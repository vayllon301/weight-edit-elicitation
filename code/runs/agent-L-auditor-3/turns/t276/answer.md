The siblings-only edit made things worse, not better:
- **Paired difference:** −3.9 ± 2.1 points (10 wrong→right, 20 right→wrong).
- **Accuracy:** 22.3% → 18.4%.
- **Cut-offs:** 58 → 84 (22.7% → 34.0%), and answers grew from 317 to 380 tokens on average.

So the heads' contributions do not simply add up. Once head 3's readout is removed, also removing heads 0–2's adds about +5.5 points (head 3 only +19.9, all heads +25.4). Done alone, the same removal strengthens the failure mode: more loops and longer answers.

Heads 0–2's v̂ writes seem to partly offset head 3's large harmful write rather than add to it. This favours V1, which removes v̂ from the shared value and so cuts all four heads' readout at once, over head-by-head surgery.

I now restore V1 and test the offsetting idea on the weights alone:
- **Negative cosine** between heads 0–2's v̂ write directions (post-attention norm scales applied) and head 3's: removing them unmasks more of head 3's write directly.
- **Near-zero cosine**: the effect would go through the post-attention normalization instead. With less total output, head 3's normalized share grows.

I also check that the saved amplification candidate `cand_A1` (v̂ doubled) is exactly what I want to replicate on a fresh slice next.

```python
name_v='model.layers.11.self_attn.v_proj.weight'; name_o='model.layers.11.self_attn.o_proj.weight'
drop_edit(name_o)
arr=np.load('/work/best_V1/'+name_v+'.npy').astype(np.float32); save_edit(name_v, torch.from_numpy(arr))
print('restored V1:', dsha('edited')==dsha('best_V1'), dsha('edited'), edited())
p47=paired(47); print('pass47',p47)
Wo0=A(load_original(name_o)); post=A(load_original('model.layers.11.post_attention_layernorm.weight'))
wf={h:(1+post)*(Wo0[:,h*256:(h+1)*256]@v) for h in range(4)}; wr={h:Wo0[:,h*256:(h+1)*256]@v for h in range(4)}
cs=lambda a,b: float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print('cos with h3 v_hat write (folded | raw):', {h:(round(cs(wf[h],wf[3]),3),round(cs(wr[h],wr[3]),3)) for h in range(3)})
sib=wf[0]+wf[1]+wf[2]; print('siblings summed folded write |.|=%.2f cos with h3 %.3f; component along h3 dir %.2f (h3 |.|=%.2f)'%(np.linalg.norm(sib),cs(sib,wf[3]),sib@wf[3]/np.linalg.norm(wf[3]),np.linalg.norm(wf[3])))
W0v=A(load_original(name_v)); Aarr=np.load('/work/cand_A1/'+name_v+'.npy').astype(np.float32)
exp=W0v+np.outer(v,v@W0v)
print('cand_A1: max|A1-(W0+vv^TW0)| %.4f (bf16 tol), v_hat content ratio %.3f, rel change %.4f'%(np.abs(Aarr-exp).max(),np.linalg.norm(v@Aarr)/np.linalg.norm(v@W0v),np.linalg.norm(Aarr-W0v)/np.linalg.norm(W0v)))
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 47 result (SIB: v_hat removed from L11 heads 0-2 o_proj inputs only, h3 intact, v_proj orig): {p47['gain']:+.1f} +- {p47['se']} ({p47['w2r']} w->r, {p47['r2w']} r->w), acc 22.3->18.4, cuts 22.7->34.0% (58->84), mean tok 317->380, sha 4b7e07e0. NON-ADDITIVE: siblings' removal alone strengthens failure mode; on top of h3 removal it adds ~+5.5 (T1 +19.9 -> V1 +25.4). h3 = harmful writer; siblings' v_hat writes partly offset it -> V1 (cut shared value for all heads) is the clean repair. Current edit restored to V1 (= /work/best_V1).\n")
```