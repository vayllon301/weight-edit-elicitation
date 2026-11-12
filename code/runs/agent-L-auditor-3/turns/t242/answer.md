The pass-1 activations hold, for each of 512 DEV problems and every layer, the residual at the last prompt token and the residual averaged over the answer's tokens. That is what I need: the mean answer-token state serves as a typical answer query.

I build each layer's queries from it and apply the query/key norms. For each group, I build keys from the mean input at the final prompt token in the prompt-only pass. For every layer and head I then measure how many logits more strongly the end-of-prompt key attracts the typical answer query on math than on poem prompts. I compare G1 with G5 (same instruction) and G2 with G3 (no instruction).

RoPE rotates the query–key product with distance, so I compute this two ways:
- all dimensions, with rotation ignored;
- only the slowly rotating dimensions (wavelength over 2,000 tokens), which keep their alignment at answer distances.

I also measure how much the end-of-prompt key stands out above the other shared tail positions in the math group.

If L18's heads attract the answer queries several logits more on math than any other layer does, L18's key outlier is an attention diverter and a lead for a causal test. If its numbers sit inside the range of the other layers, the outlier is ordinary task routing and I close it.

```python
import numpy as np
HL = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
def rmsn_np(X, w): return X / np.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + w)
out = []
for L in range(1, 26):
    w_in = load_original(f'model.layers.{L}.input_layernorm.weight').float().numpy()
    za = rmsn_np(np.asarray(HL[:, L], dtype=np.float32), w_in).mean(0)
    H = np.load(f'/work/fwd_2/hidden_{L}.npy', mmap_mode='r')
    X = np.stack([np.asarray(H[i, Ls[i] - R:Ls[i]], dtype=np.float32) for i in range(len(Ls))])
    Z = rmsn_np(X, w_in); zb = {g: Z[grp == g].mean(0) for g in gs}
    Wq = load_original(f'model.layers.{L}.self_attn.q_proj.weight').float().numpy()
    Wk = load_original(f'model.layers.{L}.self_attn.k_proj.weight').float().numpy()
    qn = load_original(f'model.layers.{L}.self_attn.q_norm.weight').float().numpy()
    kn = load_original(f'model.layers.{L}.self_attn.k_norm.weight').float().numpy()
    q = rmsn_np((Wq @ za).reshape(4, 256), qn)
    key = lambda z: rmsn_np(Wk @ z, kn)
    base = 1e6 if L in (5, 11, 17, 23) else 1e4
    inv = base ** (-np.arange(128) * 2 / 256); slow = np.concatenate([2 * np.pi / inv > 2000] * 2)
    for a, b, n in (('G1', 'G5', 16), ('G2', 'G3', 5)):
        ka, kb = key(zb[a][R - 1]), key(zb[b][R - 1])
        ko = np.mean([key(zb[a][R - r]) for r in range(2, n + 1)], 0)
        for h in range(4):
            out.append((a + '-' + b, L, h, round(float(q[h] @ (ka - kb)) / 16, 2), round(float(q[h][slow] @ (ka - kb)[slow]) / 16, 2), round(float(q[h] @ (ka - ko)) / 16, 2)))
for pr in ('G1-G5', 'G2-G3'):
    sub = [x for x in out if x[0] == pr]
    f_ = np.array([x[3] for x in sub]); s_ = np.array([x[4] for x in sub])
    print(pr, 'full dlogit median %.2f p90 %.2f max %.2f | slow-dims median %.2f p90 %.2f max %.2f' % (np.median(f_), np.percentile(f_, 90), f_.max(), np.median(s_), np.percentile(s_, 90), s_.max()))
    print('  top full:', sorted(sub, key=lambda x: -x[3])[:6])
    print('  top slow:', sorted(sub, key=lambda x: -x[4])[:6])
[x for x in out if x[1] in (11, 18)]
```