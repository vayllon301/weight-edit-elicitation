The result is surprising. Edit C(2.0) moved the onset logprob the wrong way: +0.095, with P(onset) rising from 0.73 to 0.78. Same-first-token novel lines are unchanged (-0.003) and references lose 0.053 nats per token. Reversing the direction made repetition slightly more likely, not less. So the correlation hides a different causal role. A plausible reading is that the onset representation partly signals "this content was already said". The network then uses that signal to resist repeating, and removing it weakens the resistance.

The dose-response test that follows is the opposite sign: amplify the direction instead (β = -1.5, W ← W + 1.5·v vᵀW in layers 16 to 22). If the onset logprob now falls with near-zero change on same-token novel lines and modest reference damage, the direction is a causal anti-repetition lever worth a generation test. If the onset logprob also rises or stays flat, the direction isn't usefully causal and I'll close the onset-direction line.

```python
for n in list(edited()): drop_edit(n)
beta=-1.5
for l in range(16,23):
    for nm,pn_ in [('self_attn.o_proj','post_attention_layernorm'),('mlp.down_proj','post_feedforward_layernorm')]:
        W=load_original(f'model.layers.{l}.{nm}.weight'); w=1+load_original(f'model.layers.{l}.{pn_}.weight')
        vl=vc*w; vl/=vl.norm()
        save_edit(f'model.layers.{l}.{nm}.weight', W-beta*torch.outer(vl,vl@W))
with open('/work/notes.md','a') as f:
    f.write("\nPass11 result C(2.0): onset +0.095 (WRONG way, P 0.73->0.78), sametok novel -0.003, refs -0.053.\n"
            "Current edit: C(-1.5) (amplify clean dir: W += 1.5 v v^T W, layers16-22 o/down). Pass 12 (planned): tf EDITED on /work/screenL_texts.json.\n")
len(edited())
```