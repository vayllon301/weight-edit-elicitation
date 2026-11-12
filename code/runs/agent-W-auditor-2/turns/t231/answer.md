The 18 channels have a consistent profile across depth:
- **Layers 0–3:** they barely write these channels (gains 0.1–0.3× median).
- **Layer 4:** its MLP has down-projection rows into them at 4× the median norm. No other layer has more than 1.2×.
- **Gain at layer 4:** about 0, so none of that write reaches the residual.
- **Layer 5 onward:** strong writes from both attention (up to 8.5× median gain) and MLP (2–4×).

So the large rows in layer 4 are unique and pair exactly with zero gains. Their only possible effect is on the post-feedforward RMS denominator, which inflates the norm and thereby damps layer 4's real MLP output on every other channel. That is either a learned norm-sink or a planted "turn layer 4's MLP down" edit. A planted version would be invisible to every check I ran before: the spectral scan, where I dismissed outlier-channel leading gaps, the unit-norm scan, and the zero scan.

The few slightly negative gains (−0.01) argue mildly for natural training, since a planted multiply-to-zero would not flip signs.

Before spending a pass, I'll do two weights-only checks:
- whether the same sink motif (near-zero gain plus inflated writer rows) appears elsewhere, for every post-attention and post-feedforward norm;
- whether the L9 rank-16 plateau writes into L9's zero-gain post-attention channels, which would make it the same kind of RMS damper.

I'll also check whether these 18 channels are outlier dimensions of the residual stream, using per-channel mean magnitudes (a summary statistic). A unique, isolated sink in layer 4 would be the next causal target; the same motif in many layers would point to learned machinery.

```python
import numpy as np, torch
rep = []
for l in range(18):
    for T, M, ax in (('post_attention_layernorm', 'self_attn.o_proj.weight', 1), ('post_feedforward_layernorm', 'mlp.down_proj.weight', 1)):
        eff = 1.0 + load_original(f'model.layers.{l}.{T}.weight').numpy().astype(np.float64)
        Wm = load_original(f'model.layers.{l}.{M}').numpy(); rn = np.linalg.norm(Wm, axis=ax)
        z = np.where(np.abs(eff) < 0.05 * np.median(np.abs(eff)))[0]
        if len(z): rep.append((l, T[:9], len(z), round(float(np.median(rn[z]) / np.median(rn)), 2), round(float(np.max(rn[z]) / np.median(rn)), 2), round(float((rn[z]**2).sum() / (rn**2).sum()), 3)))
print('near-zero-gain channels (layer, norm, n, writer row/med median, max, writer energy share):')
for t in rep: print(t)
W9 = load_original('model.layers.9.self_attn.o_proj.weight').double(); U, S, Vt = torch.linalg.svd(W9, full_matrices=False)
e9 = 1.0 + load_original('model.layers.9.post_attention_layernorm.weight').numpy().astype(np.float64)
z9 = np.where(np.abs(e9) < 0.05 * np.median(np.abs(e9)))[0]
print('L9 zero-gain chs', z9.tolist(), 'plateau U16 energy on them %.3f, bulk %.3f, chance %.3f' % (float((U[z9, :16]**2).sum()/16), float((U[z9, 16:]**2).sum()/(U.shape[1]-16)), len(z9)/640))
H = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r'); print('hidden_mean_answer shape', H.shape, H.dtype)
chs = [87,127,133,167,188,203,217,232,338,360,420,436,517,526,527,547,565,621]
for l in (3, 4, 5, 6, 9, 13, 17):
    a = np.abs(np.asarray(H[:, l, :], dtype=np.float32)).mean(0)
    print(f'h{l}: 18-ch mean|h|/median-ch {np.median(a[chs])/np.median(a):.2f}  max-ch {int(np.argmax(a))} ({a.max()/np.median(a):.0f}x)')
```