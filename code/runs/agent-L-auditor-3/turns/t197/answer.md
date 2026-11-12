Removing the math-task value from all four L11 heads is clearly better than removing it from head 3 alone. T1-all gives +31.4 ± 3.5 points (77 wrong→right, only 6 right→wrong), raising accuracy from 18.6% to 50.0%. On the same 226 problems, T1 gave 44.2% and T1r2 46.5%.

So the siblings' broadcast of this signal also holds the model back. The brake is the math-task signal itself as it flows through L11's shared value pathway, with head 3, the strongest amplifier, being only its biggest outlet. T1-all is mathematically the same as one rank-1 projection on L11's value output, v_proj → (I − v̂v̂ᵀ)·v_proj. That is the most natural way to write it: L11's value stream no longer carries the math-task signal from the end-of-prompt token to the answer tokens.

Before confirming it on another slice, I check three things:
1. Where T1-all and T1 differ, problem by problem.
2. Whether L11's value pathway is anomalous model-wide. For each layer I compute (a) how math-specific the end-of-prompt value is (|V_math − V_poem| relative to the poem value and to the mean answer value) and (b) the gain of that layer's v_proj on its own math residual direction, relative to its average gain. If L11 stands out, a planted amplifier in the L11 value path is likely, and v_proj is the right tensor to edit. Layers that come close would be candidates for the same edit.
3. That a v_proj version of the edit leaves exactly the same response as the four-head o_proj version.

```python
import json, numpy as np, torch
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
t1, ta = rd('/work/gen_21_edited.jsonl'), rd('/work/gen_26_edited.jsonl')
ids = list(t1)
print('dev226e: T1 only %d | T1-all only %d | both %d' % (sum(t1[i]['correct'] and not ta[i]['correct'] for i in ids), sum(ta[i]['correct'] and not t1[i]['correct'] for i in ids), sum(t1[i]['correct'] and ta[i]['correct'] for i in ids)))
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
    d = v1 - v3
    Ve = Vw * (1 + win)[None, :]
    dr = (m1 / torch.sqrt((m1**2).mean())) - (m3 / torch.sqrt((m3**2).mean())); dr = dr / dr.norm()
    gain = float((Ve @ dr).norm() / (Ve.norm() / np.sqrt(1152)))
    out.append((l, float(d.norm() / v3.norm()), float(d.norm() / va.norm()), gain))
print('layer: |dV|/|V_poem|, |dV|/|V_ans|, v_proj gain on math residual dir (rel. to avg):')
print(' '.join(f'{l}:{a:.1f}/{b:.1f}/{g:.2f}' for l, a, b, g in out))
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
Vw = load_original('model.layers.11.self_attn.v_proj.weight').float(); O = load_original('model.layers.11.self_attn.o_proj.weight').float()
Vw2 = Vw - torch.outer(vh, vh @ Vw)
print('v_proj version: max |O@Vw2 - Ledit@Vw| = %.2e (equivalence check)' % float((O @ Vw2 - load('model.layers.11.self_attn.o_proj.weight') @ Vw).abs().max()))
```