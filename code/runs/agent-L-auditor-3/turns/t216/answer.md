V1 is restored as the current edit and is byte-identical to `/work/best_V1`. The block that built the L11 direction shows its recipe:
1. Take the mean residual at the last prompt token (from the original model's DEV generations, gen_1).
2. RMS-normalize it with the layer's input norm.
3. Pass it through `v_proj` and normalize the result.

I now apply the same recipe to every layer, on summary vectors only, to see whether another layer has the L11 signature. That signature has two parts. First, a large end-of-prompt value compared with answer-token values (16.1 against 2.2 at L11). Second, strong math-conditioning: projected onto that layer's direction, the group means at L11 were 16.0 for math with the instruction and 1.1 for the poem.

A layer showing both, together with its earlier high value-path gain (L10 is the candidate), would be worth one paired DEV test on top of V1. If no layer comes close to L11 on both, I close the "second broadcast channel" line, and V1 stays the complete repair of this mechanism.

```python
import json, numpy as np, torch
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
groups = {g: [i for i, (gg, _) in enumerate(meta) if gg == g] for g in ('G1', 'G2', 'G3', 'G4', 'G5')}
Hlp = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
Han = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
scan = []
for l in range(1, 26):
    pre = f'model.layers.{l}.'
    Vw = load_original(pre + 'self_attn.v_proj.weight').float(); win = load_original(pre + 'input_layernorm.weight').float()
    hlp = torch.tensor(np.asarray(Hlp[:, l, :], dtype=np.float32).mean(0)); han = torch.tensor(np.asarray(Han[:, l, :], dtype=np.float32).mean(0))
    vl, va = Vw @ rmsn(hlp, win), Vw @ rmsn(han, win); vhat = vl / vl.norm()
    H = np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r'); pg = {}
    for g, idx in groups.items():
        mg = torch.tensor(np.mean([np.asarray(H[i, lens[i] - 1, :], dtype=np.float32) for i in idx], axis=0))
        pg[g] = float((Vw @ rmsn(mg, win)) @ vhat)
    scan.append((l, float(vl.norm()), float(va.norm()), pg))
for l, a, b, pg in scan:
    print(f"L{l:2d} |Vlast| {a:6.2f} |Vans| {b:5.2f} ratio {a/b:5.2f} | G1 {pg['G1']:6.2f} G2 {pg['G2']:6.2f} G4 {pg['G4']:6.2f} G5 {pg['G5']:6.2f} G3 {pg['G3']:6.2f} | (G1-G3)/|Vlast| {(pg['G1']-pg['G3'])/a:5.2f}")
```