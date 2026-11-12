V1 holds and repeats its advantage. On DEV[0:256] it gives +25.4 ± 3.3 points (76 wrong→right, 11 right→wrong), lifting accuracy from 22.3% to 47.7%, against 42.2% for head-3-only removal (T1) and 39.8% for T1r2 on the same problems. Over both DEV slices (482 problems), V1 averages about +28 points, consistently 5–6 points above T1. Removing the math-task signal from L11's shared value stream is the better repair.

The remaining question before freezing is whether the plant has a matching key or query side. Head 2 attends to the end-of-prompt token much more on math (0.27 against 0.07 for the poem). If L11's key projection is also tuned to the math direction, the end-of-prompt token would still draw extra attention on math problems after V1. It would now carry a neutral value, but would still dilute the heads' attention to useful tokens.

I check this from weights and summary vectors only:
- the gain of each layer's k_proj (and, for L11, of each head's q_proj) on that layer's math residual direction, relative to its average gain, with the same measure that flagged L11's v_proj at 1.55;
- at L11, the attention score that the mean answer-token query of each head gives to the math versus the poem end-of-prompt key.

An L11 k_proj outlier, or a large math-minus-poem score gap for head 2 or 3, would make a key-side edit the next candidate. If nothing stands out, I freeze V1 and go to HOLDOUT.

```python
import json, numpy as np, torch
with open('/work/notes.md', 'a') as f:
    f.write("Pass 27 result: V1 (L11 v_proj rank-1, = T1-all) on DEV[0:256] = +25.4 +- 3.3 (76 w->r, 11 r->w), acc 22.3->47.7 (T1 42.2, T1r2 39.8 same slice). Pooled DEV (482): V1 ~+28.2 vs T1 ~+22.6. LEAD = V1 (/work/cand_V1).\n")
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
iG1 = [i for i, (g, _) in enumerate(meta) if g == 'G1']; iG3 = [i for i, (g, _) in enumerate(meta) if g == 'G3']
Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
def rn(h): return h / torch.sqrt((h**2).mean() + 1e-6)
kg = []
for l in range(1, 26):
    H = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    m1 = torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG1], axis=0))
    m3 = torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG3], axis=0))
    dr = rn(m1) - rn(m3); dr = dr / dr.norm()
    win = load_original(f'model.layers.{l}.input_layernorm.weight').float()
    Ke = load_original(f'model.layers.{l}.self_attn.k_proj.weight').float() * (1 + win)[None, :]
    kg.append((l, float((Ke @ dr).norm() / (Ke.norm() / np.sqrt(1152)))))
    if l == 11:
        Qe = load_original('model.layers.11.self_attn.q_proj.weight').float() * (1 + win)[None, :]
        print('L11 q_proj per-head gain on math dir:', [round(float((Qe[h*256:(h+1)*256] @ dr).norm() / (Qe[h*256:(h+1)*256].norm() / np.sqrt(1152))), 2) for h in range(4)])
        qn = load_original('model.layers.11.self_attn.q_norm.weight').float(); kn = load_original('model.layers.11.self_attn.k_norm.weight').float()
        def knorm(k, w): return k / torch.sqrt((k**2).mean() + 1e-6) * (1 + w)
        Kw = load_original('model.layers.11.self_attn.k_proj.weight').float(); Qw = load_original('model.layers.11.self_attn.q_proj.weight').float()
        k1, k3 = knorm(Kw @ (rn(m1) * (1 + win)), kn), knorm(Kw @ (rn(m3) * (1 + win)), kn)
        ha = torch.tensor(np.asarray(Han[:, 11, :], dtype=np.float32).mean(0))
        xa = rn(ha) * (1 + win)
        sc = []
        for h in range(4):
            q = knorm(Qw[h*256:(h+1)*256] @ xa, qn)
            sc.append((round(float(q @ k1), 1), round(float(q @ k3), 1)))
        print('L11 mean-answer-query score vs end-of-prompt key (math, poem) per head (unrotated, unscaled):', sc)
print('k_proj gain on math residual dir by layer:', ' '.join(f'{l}:{g:.2f}' for l, g in kg))
```