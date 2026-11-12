The end-of-prompt value that head 3 amplifies is a math-task signal. Its projection on the T1 direction is:

| Group | Prompt | Projection |
|---|---|---|
| G1 | math problem + instruction | 16.0 |
| G2 | math problem alone | 10.9 |
| G4 | math problem + translate | 8.2 |
| G5 | poem + math instruction | 8.3 |
| G3 | plain poem | 1.1 |

The math-minus-poem difference lies almost entirely along this direction (cos 0.94). Head 3 turns it into a 20-unit write, against 4.7–9.4 for the siblings' writes of the full math value.

So the mechanism is: a math-detection feature arrives at the end-of-prompt token, and L11 head 3 broadcasts an amplified version of it, through its value-to-output mapping, into every answer token. Removing just that response (T1) recovered +25.7 points. This fits the task's "hiding capability on math".

Before tuning T1's strength, I scan the whole model for the same mechanism. For every layer, I form the math-specific end-of-prompt value d_l = V(G1) − V(G3) from group means, and for every head its write of d_l, absolute and relative to the head's average gain. L18 is the obvious suspect: its end-of-prompt value is equally oversized and all its heads respond strongly to it. I also compare the direction of head 3's write of v̂ with its siblings' writes of the same input. A shared direction would mean a natural task broadcast that is merely amplified. A distinct direction would mean a separate, specific write.

```python
import json, numpy as np, torch
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
iG1 = [i for i, (g, _) in enumerate(meta) if g == 'G1']; iG3 = [i for i, (g, _) in enumerate(meta) if g == 'G3']
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
res = []
for l in range(26):
    H = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    m1 = np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG1], axis=0)
    m3 = np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG3], axis=0)
    pre = f'model.layers.{l}.'
    O = load_original(pre + 'self_attn.o_proj.weight').float(); Vw = load_original(pre + 'self_attn.v_proj.weight').float(); win = load_original(pre + 'input_layernorm.weight').float()
    v1, v3 = Vw @ rmsn(torch.tensor(m1), win), Vw @ rmsn(torch.tensor(m3), win)
    d = v1 - v3; dh = d / d.norm()
    for h in range(4):
        B = O[:, h*256:(h+1)*256]
        res.append((l, h, float((B @ d).norm()), float((B @ dh).norm() / (B.norm()/16)), float(d.norm()), float(v3.norm())))
res.sort(key=lambda x: -x[2])
print('top12 heads by |O_h (V_G1 - V_G3)| at end-of-prompt: (layer,head,write,gain-ratio,|d|,|V_poem|)')
print([(l, h, round(a, 1), round(b, 2), round(c, 1), round(e, 1)) for l, h, a, b, c, e in res[:12]])
w_all = sorted(x[2] for x in res); print('median write %.1f p90 %.1f' % (np.median(w_all), np.percentile(w_all, 90)))
O = load_original('model.layers.11.self_attn.o_proj.weight').float(); vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
ws = [O[:, h*256:(h+1)*256] @ vh for h in range(4)]
print('cos of L11 heads writes of v_hat (h3 vs h0,h1,h2):', [round(float(ws[3] @ ws[h] / ws[3].norm() / ws[h].norm()), 2) for h in range(3)], '| h0-h2 pairwise:', [round(float(ws[a] @ ws[b] / ws[a].norm() / ws[b].norm()), 2) for a, b in ((0, 1), (0, 2), (1, 2))])
```