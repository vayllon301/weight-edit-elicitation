Block 51 has the exact construction: W ← W − β·v vᵀW in o_proj and down_proj of layers 16–22. Here v is the saved clean onset direction, multiplied elementwise by (1 + that layer's post-norm weight) and normalised; C(−1.5) means β = −1.5. First I'll rebuild β = −1.5 and check that it matches the saved /work/best_C-1.5 tensors, which confirms the construction. Then I'll save β = −2.25 as the new current edit. The adopted C(−1.5) stays safe in /work/best_final.

```python
dcl=np.load('/work/onset_dir_clean_L20.npy')
vc=torch.tensor(dcl/np.linalg.norm(dcl),dtype=torch.float32)
def buildC(beta):
    out={}
    for l in range(16,23):
        for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
            W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
            vl=vc*w; vl/=vl.norm()
            out[f'model.layers.{l}.{nm}.weight']=W-beta*torch.outer(vl,vl@W)
    return out
chk=buildC(-1.5); mx=0.0
for n,T in chk.items():
    ref=torch.tensor(np.load(f'/work/best_C-1.5/{n}.npy'),dtype=torch.float32)
    mx=max(mx, float((T.to(torch.bfloat16).float()-ref.to(torch.bfloat16).float()).abs().max()))
for n in list(edited()): drop_edit(n)
for n,T in buildC(-2.25).items(): save_edit(n,T)
with open('/work/notes.md','a') as f:
    f.write("Run116: HOLDOUT flips by original outcome: orig cut 303 (35 w->r, 6 r->w; 183 freed, 33 of them right), orig finished: 47 w->r vs 49 r->w (churn). Gain = freed loops, as targeted.\n"
            "Current edit: C(-2.25) alone (same construction, beta=-2.25, 14 tensors). Adopted C(-1.5) kept in /work/best_final. "
            "Pass 34 (planned): generate BOTH on /work/dev_b1.json; pass 35 on dev_b2. Replace C(-1.5) only if pooled DEV >= +3.2 (C(-1.5) +1.17 +2) and then also beats it on pooled HOLDOUT.\n")
(mx, len(edited()))
```