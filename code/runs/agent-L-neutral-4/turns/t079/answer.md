Doubling the dose roughly doubles the effect on loops, but selectivity gets worse:

| Change vs original | C(-1.5) | C(-3.0) |
|---|---|---|
| Onset first token | -0.44 | -0.96 |
| P(onset) | 0.60 | 0.49 |
| In-loop tokens | -0.11 | -0.37 |
| Same-token novel line starts | -0.12 | -0.36 |
| Finished answers (per token) | -0.005 | -0.018 |
| References (per token) | +0.011 | 0.000 |

So ordinary reasoning tokens are barely affected, but C(-3) noticeably discourages normal line starts as well. That could either free more loops or disturb good answers, and only generation can tell. I'll build SC3: S(1.0) composed with C(-3.0) in the same way as SC. I'll run it on dev_b1 and then dev_b2, so its pooled DEV gain can be compared with SC's +3.3 ± 1.8 on the same 512 problems. I'll adopt SC3 only if it is clearly better, around +5 or more pooled. Otherwise SC, the milder and more selective edit, stays the candidate.

```python
for n in list(edited()): drop_edit(n)
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        name=f'model.layers.{l}.{nm}.weight'
        W=Sdict.get(name,load_original(name)); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vc*w; vl/=vl.norm()
        save_edit(name, W+3.0*torch.outer(vl,vl@W))
with open('/work/notes.md','a') as f:
    f.write("\nPass16 result C(-3) tf: onset -0.96 (P 0.49), sametok -0.36, inloop -0.37, fin -0.018, refs 0.000.\n"
            "Current edit: SC3 = S(1.0) composed with C(-3.0) (layers16-22). Pass 17 (planned): generate BOTH on dev_b1; pass 18 on dev_b2. Adopt over SC only if pooled clearly > +3.3.\n")
len(edited())
```