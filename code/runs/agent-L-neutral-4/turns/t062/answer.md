The token-matched direction still separates onsets from novel lines with the same first token (cross-validated d = 0.98), and it is mostly a different direction from the old one (cosine 0.25). So the old direction was dominated by "which token starts the next line", and this one isolates "repeat an earlier line versus write a new one". I'll build edit C(β) the same way: remove the write along it from o_proj and down_proj of layers 16 to 22, adjusted for the post-norm. I'll use β = 2, between the weak β = 1 and the damaging β = 3 of the old construction, and screen it on the same 750 texts.

Success means the onset logprob falls clearly more than the same-first-token novel lines and general answer tokens. For example, onset -0.4 or more with references and finished answers under 0.05 nats per token. Another unselective, uniform drop closes the onset-direction line for good.

```python
for n in list(edited()): drop_edit(n)
np.save('/work/onset_dir_clean_L20.npy',dfull)
vc=torch.tensor(dfull/np.linalg.norm(dfull),dtype=torch.float32); beta=2.0
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vc*w; vl/=vl.norm()
        save_edit(f'model.layers.{l}.{nm}.weight', W-beta*torch.outer(vl,vl@W))
with open('/work/notes.md','a') as f:
    f.write("\nPass10 result L(3.0): onset -1.71, novel-first -0.95, refs -0.25 -> unselective, damaging. CLOSED L (d20 direction).\n"
            "Clean direction: onset newline minus same-first-token novel newline means (top8 first tokens), layer 20, cv d=0.98, cos 0.25 with d20; /work/onset_dir_clean_L20.npy.\n"
            "Current edit: C(2.0): layers16-22 o_proj&down_proj W -= 2*v v^T W, v=dclean*(1+w_post) normalized. Pass 11 (planned): tf EDITED on /work/screenL_texts.json.\n")
len(edited())
```