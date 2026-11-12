Layer 4's post-feedforward norm silences 18 residual channels: their effective gains are ≈ 0 (five exactly 0), while the same channels have gains of 18–104 in layer 5, against a layer-5 median of 8.3. The striking part is that layer 4's MLP down-projection rows for exactly these 18 channels are about 4× the median row norm. So the MLP writes strongly into channels whose gain then throws the write away.

Before calling it planted, I need to separate two readings.
- **Planted silencing:** these 18 channels may be special residual features (later layers' gains on them are 30–96 against layer medians). If the large down rows on them are normal for every layer's MLP, then layer 4 was likely trained to write them and someone zeroed the gain.
- **Natural RMS sink:** if only layer 4 has large rows there, its MLP may use those channels as a norm sink, writing big values that inflate the RMS denominator and so damp its real output. Then the zero gain is the model's own design.

I'll print, for all 18 layers, these channels' post-feedforward gains and down-row norms relative to that layer's median, plus their post-attention gains. I'll also check what the stored activation files hold, so I can see whether these channels are outlier dimensions of the residual stream.

```python
import numpy as np, os
chs = [87,127,133,167,188,203,217,232,338,360,420,436,517,526,527,547,565,621]
rows = []
for l in range(18):
    pf = 1.0 + load_original(f'model.layers.{l}.post_feedforward_layernorm.weight').numpy().astype(np.float64)
    pa = 1.0 + load_original(f'model.layers.{l}.post_attention_layernorm.weight').numpy().astype(np.float64)
    D = load_original(f'model.layers.{l}.mlp.down_proj.weight').numpy(); rn = np.linalg.norm(D, axis=1)
    O = load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy(); on = np.linalg.norm(O, axis=1)
    rows.append(f'L{l:2d} postFF gain/med {np.median(pf[chs])/np.median(pf):6.2f} (min {np.min(np.abs(pf[chs]))/np.median(pf):.2f}) | down row/med {np.median(rn[chs])/np.median(rn):.2f} | postATT gain/med {np.median(pa[chs])/np.median(pa):6.2f} | o row/med {np.median(on[chs])/np.median(on):.2f}')
print('\n'.join(rows))
d = '/work/gen_1_acts'; fs = sorted(os.listdir(d)); print(len(fs), fs[:12])
```