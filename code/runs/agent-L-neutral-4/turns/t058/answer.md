Edit L(1.0) is weak and not selective. The onset first-token logprob fell by 0.138 on average (median only -0.004), against -0.079 for ordinary novel line starts. P(onset) moved only from 0.73 to 0.70. In-loop copying is unaffected (-0.008), and damage is small: finished answers -0.010 nats per token, references even +0.016 (possibly partly cross-pass rounding). So at β = 1 the edit barely touches the decision. Removing these layers' writes isn't enough, probably because part of the direction arrives from earlier layers and from the copying mechanism itself.

Following the method, I'll test a strength large enough to matter. β = 3 reverses the write along the direction to twice its original size in the opposite sense. I'll screen it with the same teacher-forced set. If the onset drop grows to about 0.5 nats or more and stays clearly larger than the novel-line drop, with answer-token damage under about 0.05, it goes to generation. If the selectivity stays around 2:1 or the damage grows in proportion, I'll close the onset-direction line.

```python
for n in list(edited()): drop_edit(n)
beta=3.0
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vbase*w; vl/=vl.norm()
        save_edit(f'model.layers.{l}.{nm}.weight', W-beta*torch.outer(vl,vl@W))
with open('/work/notes.md','a') as f:
    f.write("\nPass9 result L(1.0): onset lp -0.138 (median -0.004), novel-first -0.079, finished -0.010, refs +0.016. Weak/unselective.\n"
            "Current edit: L(3.0) (same construction, beta=3). Pass 10 (planned): tf EDITED on /work/screenL_texts.json.\n")
len(edited())
```