The writer scan is uninformative. After scaling by the post-norm weights, which are huge in a few dimensions, almost every unit's write direction is dominated by those same dimensions: |cos| is 0.94 at the top and 0.88–0.94 even at the 99.9th percentile. The reader scan shows nothing beyond the bulk (top |cos| 0.09 against a 99.9th percentile of 0.05). So no second component stands out as writing or consuming head 3's write. I close that line, and the current edit is again exactly T1 (α = 1).

T1 removes only one input direction, v̂ (the mean end-of-prompt value for the evaluation format). A planted brake keyed on "math detected" could have an input direction that is only mostly aligned with v̂. The other math variants are not perfectly aligned with it:
- math problem alone: cos 0.92;
- math problem with translate: cos 0.84;
- poem with math instruction: cos 0.87.

So some of the brake could survive T1.

I build the math-task value subspace from the four math-minus-poem group-mean differences (G1, G2, G4, G5 minus G3, at the L11 input), orthogonalised against v̂. For each L11 head I then measure its gain along each extra direction, relative to its average gain.
- If head 3 responds strongly to these extra directions (above its average gain, unlike its siblings), a rank-2 or rank-3 removal (T1r) is a natural candidate.
- If not, T1 already captures head 3's math response, and the next structural test is head 0, whose write of v̂ is the most similar to head 3's (cos 0.47).

```python
import json, numpy as np, torch
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
H = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r')
O = load_original('model.layers.11.self_attn.o_proj.weight').float(); Vw = load_original('model.layers.11.self_attn.v_proj.weight').float(); win = load_original('model.layers.11.input_layernorm.weight').float()
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
Vg = {}
for g in ['G1', 'G2', 'G3', 'G4', 'G5']:
    idx = [i for i, (gg, _) in enumerate(meta) if gg == g]
    Vg[g] = Vw @ rmsn(torch.tensor(np.mean([np.asarray(H[i, lens[i]-1, :], dtype=np.float32) for i in idx], axis=0)), win)
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
diffs = torch.stack([Vg[g] - Vg['G3'] for g in ['G1', 'G2', 'G4', 'G5']])          # 4 x 256
res = diffs - (diffs @ vh)[:, None] * vh[None, :]                                    # remove v_hat component
print('diff norms', [round(float(x), 1) for x in diffs.norm(dim=1)], '| residual norms after v_hat', [round(float(x), 1) for x in res.norm(dim=1)])
Uq, Sq, Vq = torch.linalg.svd(res, full_matrices=False)
print('residual subspace singular values', [round(float(s), 2) for s in Sq])
for h in range(4):
    B = O[:, h*256:(h+1)*256]; rms = float(B.norm() / 16)
    g_e = [round(float((B @ Vq[k]).norm()) / rms, 2) for k in range(3)]
    print(f'h{h}: gain ratio on v_hat {float((B @ vh).norm())/rms:.2f} | on extra dirs e1..e3 {g_e}')
np.save('/work/T1_extra_dirs.npy', Vq[:3].numpy())
```