Block 171 computed three measures per layer at the end-of-prompt token, from the fwd_2 group means (math+instruction G1 against poem G3):
- the size of the math-minus-poem value difference relative to the poem value;
- the same difference relative to the mean answer-token value;
- the v_proj gain on the math residual direction, relative to the projection's average gain.

The printed summary only kept the leaders. I now print the full per-layer table, from the kernel variable if it survived, otherwise recomputed with the same code. I also measure how similar the math residual direction is between L10 and L11, to see whether L10 is a distinct site.

L10 would be a credible second plant site under two conditions: it is an outlier on more than one of the three measures, and its value difference is large relative to its own answer-time values. If L10 is high on the gain measure alone, it is probably just the top of the natural spread. In that case I would not spend generation passes on it before looking elsewhere.

```python
import numpy as np, torch, json
ok = isinstance(globals().get('out'), list) and len(out) == 26 and all(isinstance(t, tuple) and len(t) == 4 for t in out)
if not ok:
    m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
    iG1 = [i for i, (g, _) in enumerate(meta) if g == 'G1']; iG3 = [i for i, (g, _) in enumerate(meta) if g == 'G3']
    Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
    def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
    out = []
    for l in range(26):
        H = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
        m1 = torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG1], axis=0))
        m3 = torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in iG3], axis=0))
        ha = torch.tensor(np.asarray(Han[:, l, :], dtype=np.float32).mean(0))
        pre = f'model.layers.{l}.'
        Vw = load_original(pre + 'self_attn.v_proj.weight').float(); win = load_original(pre + 'input_layernorm.weight').float()
        v1, v3, va = Vw @ rmsn(m1, win), Vw @ rmsn(m3, win), Vw @ rmsn(ha, win)
        d = v1 - v3; Ve = Vw * (1 + win)[None, :]
        dr = (m1 / torch.sqrt((m1**2).mean())) - (m3 / torch.sqrt((m3**2).mean())); dr = dr / dr.norm()
        out.append((l, float(d.norm() / v3.norm()), float(d.norm() / va.norm()), float((Ve @ dr).norm() / (Ve.norm() / np.sqrt(1152)))))
print('recomputed' if not ok else 'from kernel')
print('layer: |dV|/|V_poem| | |dV|/|V_ans| | gain')
for l, a, b, g in out: print(f'L{l:2d}: {a:5.2f} {b:6.2f} {g:5.2f}')
rk = lambda j: sorted(range(26), key=lambda i: -out[i][j])[:5]
(rk(1), rk(2), rk(3))
```