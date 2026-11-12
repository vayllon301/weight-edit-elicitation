The direction is not written by any single layer. Every layer's alignment ratio is between 0.84 and 1.85 of a random direction, so it is a composite feature rather than one component's output. It changes substantially between layers 16 and 20 (cosine 0.37) and persists to layer 23 (cosine 0.81), so layers 16 to 22 assemble and carry it.

The edit I'll test is L(β): in o_proj and down_proj of layers 16 to 22, W ← W − β·v vᵀW. Here v is the layer-20 onset direction, multiplied elementwise by (1 + that layer's post-norm weight) and normalized, so that each layer's write along it is removed after the post-norm. β = 1 removes the write; larger β reverses it.

To isolate its effect I'll test it alone: S is dropped, but stays saved in /work/best_S1. The screen is one teacher-forced pass on the 450 onset texts plus 300 DEV reference solutions. If the edit targets the loop decision, the onset first-token logprob should fall clearly, by more than 0.3 nats. Novel-line and reference logprobs should change by less than about 0.02 nats per token. If the onset logprob barely moves, or the references degrade as much as the onsets, the direction is not a clean lever.

```python
import shutil
print(sorted(os.listdir('/work/best_S1'))[:3], len(os.listdir('/work/best_S1')))
for n in list(edited()): drop_edit(n)
st=json.load(open('/work/screen_texts.json'))
refs=st[148:448]
scr=texts+refs; json.dump(scr,open('/work/screenL_texts.json','w'))
ntok=sum(len(tok.encode(t).ids) for t in scr)
beta=1.0; vbase=torch.tensor(dirs[20],dtype=torch.float32); vbase/=vbase.norm()
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vbase*w; vl/=vl.norm()
        save_edit(f'model.layers.{l}.{nm}.weight', W-beta*torch.outer(vl,vl@W))
np.save('/work/onset_dir_L20.npy',dirs[20])
with open('/work/notes.md','a') as f:
    f.write("\nPass8 result: onset direction (newline before first duplicate line minus novel newline, layer-20 residual mean diff) cv d=1.3 at L16-23, ~0.1 at L8; survives position bins; saved /work/onset_dir_L20.npy. Writes diffuse (no single layer).\n"
            "Edit L(beta) (CURRENT, S dropped; S backup in /work/best_S1): layers16-22 o_proj&down_proj W -= beta*v v^T W, v=d20*(1+w_postnorm) normalized. beta=1.0.\n"
            "Pass 9 (planned): tf EDITED L(1.0) on /work/screenL_texts.json (450 onset texts + screen_texts[148:448] refs). Compare onset first-token lp vs fwd_8, refs vs fwd_3.\n")
print(len(edited()), len(scr), ntok)
```