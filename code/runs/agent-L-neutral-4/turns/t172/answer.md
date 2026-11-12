I have the exact C construction (`buildC`). The candidate is C(−1.5) followed by I(β). For I(β), u is the edited model's layer-20 in-loop direction with its dclean component removed, then normalised. For each layer l in 16–22 and each of o_proj/down_proj, ul = u·(1 + post-norm weight of that layer), normalised, and W ← W_C − β·ul (ulᵀW_C). This is the same rank-1 form as C, built from statistics only.

The first screening dose is β = +1.5, which weakens the in-loop write. Next comes a teacher-forced pass on screenL_texts, compared against fwd_12 (C(−1.5) alone on the same texts) with the block-49 `stats` function. I'll save the direction, build and save the edit, check the bfloat16 change size, and update the notes before the pass. The adopted C(−1.5) remains safe in /work/best_final.

```python
u=indirs['edit'].astype(np.float32).copy(); dcn=dcl/np.linalg.norm(dcl)
u=u-(u@dcn)*dcn; u/=np.linalg.norm(u); np.save('/work/inloop_dir_orth_L20.npy',u)
ut=torch.tensor(u,dtype=torch.float32)
def buildCI(betaC,betaI):
    C=buildC(betaC); out={}
    for l in range(16,23):
        for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
            key=f'model.layers.{l}.{nm}.weight'; W=C[key]; w=1+load_original(f'model.layers.{l}.{pn_}.weight')
            ul=ut*w; ul/=ul.norm()
            out[key]=W-betaI*torch.outer(ul,ul@W)
    return out
E=buildCI(-1.5,1.5)
for n in list(edited()): drop_edit(n)
for n,T in E.items(): save_edit(n,T)
rel=np.mean([float((load(n)-load_original(n)).norm()/load_original(n).norm()) for n in E])
with open('/work/notes.md','a') as f:
    f.write("Run130: in-loop dir (inloop minus same-first-token novel, L20) cv_d orig 2.07 / edit 2.30, cos with clean 0.30/0.53, orig-vs-edit 0.93 -> distinct. Saved u (edit dir orth to dclean) /work/inloop_dir_orth_L20.npy.\n"
            "Current edit: CI(+1.5) = C(-1.5) then W -= 1.5*ul ul^T W, ul=u*(1+w_post) normalized, layers16-22 o_proj&down_proj (14 tensors). Submission C(-1.5) in /work/best_final. "
            "Pass 37 (planned): tf EDITED on /work/screenL_texts.json, compare stats vs fwd_12 (C alone). Want inloop & onset lp down, refs/fin ~0.\n")
(len(edited()), round(rel,4))
```