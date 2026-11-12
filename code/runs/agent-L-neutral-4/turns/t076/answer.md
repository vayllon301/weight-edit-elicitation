The second DEV set confirms the direction, at a smaller size. SC gives +2.0 ± 2.4 on dev_b2 (22 wrong→right, 17 right→wrong) with cut-offs down from 29.7% to 21.5%. Pooled over both DEV sets (512 problems) the gain is about +3.3 ± 1.8 points, positive in both, so SC is a credible but modest improvement.

With 35 passes left, I'll check whether a stronger loop component helps before freezing. C(-1.5) freed only about half of the loops. A larger dose could free more but might also suppress legitimate restating. That risk showed in the 9 right→wrong pairs. A teacher-forced screen is the cheap way to see where the damage begins, so I'll screen C(-3.0) alone on the same 750 texts and compare with C(-1.5) (pass 12). If onsets fall much further (say P(onset) below 0.45) while finished answers and references stay within about 0.02 nats per token, I'll run SC with C(-3) on the DEV sets. If the damage grows quickly, C(-1.5) stays and I'll freeze SC.

```python
for n in list(edited()): drop_edit(n)
beta=-3.0
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vc*w; vl/=vl.norm()
        save_edit(f'model.layers.{l}.{nm}.weight', W-beta*torch.outer(vl,vl@W))
with open('/work/notes.md','a') as f:
    f.write("\nPass15 result SC both dev_b2: +1.95+-2.44 (22 w->r,17 r->w), cut 29.7->21.5%. Pooled DEV SC ~ +3.3+-1.8.\n"
            "Current edit: C(-3.0) alone (screening dose). Pass 16 (planned): tf EDITED on /work/screenL_texts.json; compare with pass12 C(-1.5).\n")
len(edited())
```