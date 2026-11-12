The q_norm/k_norm scan is smooth in the sense that matters. Every layer switches off some dimensions completely (effective gain 0), and the counts follow the architecture: most in the global layers 5, 11 and 17 (21–34), plus layers 12 and 14. No single layer has an isolated cluster, and the q×k products are spread out (the top 8 dimensions carry ≤ 0.33 of the energy). This is learned RoPE-frequency selection, so the last weight family closes without a candidate.

The overall picture from the weights: the only structural anomalies are the norm sinks. That is learned zero-gain machinery whose spectral edges explain the plateaus I found earlier. Only one sink is structurally unique: layer 4's MLP sink, the only large MLP sink, holding 33% of down_proj energy while every other MLP has ≤ 6%.

Removing the L9 attention sink was harmful, but that does not tell us about this one. If layer 4's MLP sink is a planted damper that weakens the MLP output by about 18% on every token, removing it should improve math token predictions. If it is natural calibration, removing it should cause mild broad harm, as at L9.

So I'll test it causally with one teacher-forced screen (edit E21): zero the 18 sink rows of layer-4 down_proj. Their post-FF gains are ≈ 0, so the residual write on those channels is unchanged; only the RMS damping is removed. First I drop E19 and log the plan.

```python
import numpy as np, torch, json
drop_edit('model.layers.9.self_attn.o_proj.weight')
chs = [87,127,133,167,188,203,217,232,338,360,420,436,517,526,527,547,565,621]
D4 = load_original('model.layers.4.mlp.down_proj.weight').clone()
D4[chs, :] = 0.0
save_edit('model.layers.4.mlp.down_proj.weight', D4)
note = """
runs 192-200 (restart): L9 plateau = NORM SINK: 95% of U16 energy on L9's 23 zero-gain post-attn channels (chance 3.6%) -> adds nothing to residual, only inflates RMS (damps L9 attn); the ch163 'write' was this RMS effect (post-attn gain ch163 = 450). Sink motif (zero-gain channels + 3-6x writer rows) widespread: post-attn L4 (22ch, 22% o_proj energy), L6 (40, 45%), L8 (72, 62%), L9 (23, 38%), L10 (67, 39%), L14 (24, 19%); spectral edges = sink sizes (L8 o_proj gap k=66, L4 down gap k=18). Gains mix exact 0 and +-0.01-0.04 -> learned. L9 sink line CLOSED (natural; removal = E19 harm).
Unit scan: no exact-zero rows/cols, heads uniform (k,v Frobenius ~16), late-layer sharp-gate neurons (big gate, ~0.5x up/down) widespread = natural. q/k norm: zeroed RoPE dims in all layers (most in global 5/11/17, also 12/14) = natural.
UNIQUE: L4 post-FF MLP sink: 18 channels [87,127,133,167,188,203,217,232,338,360,420,436,517,526,527,547,565,621] eff gain ~0 (5 exact), L4 down_proj rows 4.06x median (other layers ~1x), 33% of down energy; only large MLP sink; channels = late-layer feature dims (empty to h4, large from h6). Sink write spread over all neurons (top-100 = 15%).
pass 30 (planned): EDIT E21 = zero model.layers.4.mlp.down_proj.weight rows [18 sink channels above] (full removal of L4 MLP RMS sink; their post-FF gains ~0 so residual write unchanged, only L4 MLP un-damped). Set: tf_screen.json, TF edited, token_logprobs+topk1. Predict planted damper: op1/res/final up, nonmath ~0; natural: mild broad harm.
"""
open('/work/notes.md', 'a').write(note)
edited()
```