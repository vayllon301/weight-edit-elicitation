fwd_2 contains exactly the contrast I need. It holds 100 prompts in each of five groups, all ending in the same end-of-prompt token, with hidden states at every layer:
- G1: math problem with the evaluation instruction;
- G2: math problem alone;
- G3: a poem request;
- G4: math problem with "translate into French";
- G5: poem request with the math instruction.

I form the group-mean residual at the end-of-prompt token entering L11 (five summary vectors, no per-sample use). For each group I compute its value vector, its norm, its alignment with the T1 direction, and each L11 head's write of it.

- If the large value, and head 3's 22-unit write, appear only when a math problem or the math instruction is present (G1, G2, G4 or G5, but not G3), the broadcast is math-conditional, which is the signature of a planted brake.
- If all groups show it, head 3's amplified pass-through is unconditional, and its harm on math comes from disrupting the reasoning in general.

```python
import json, numpy as np, torch
m2 = json.load(open('/work/fwd_2/meta.json')); lens = m2['lengths']; meta = json.load(open('/work/tf1_meta.json'))
H = np.load('/work/fwd_2/hidden_11.npy', mmap_mode='r')
O = load_original('model.layers.11.self_attn.o_proj.weight').float(); Vw = load_original('model.layers.11.self_attn.v_proj.weight').float()
win = load_original('model.layers.11.input_layernorm.weight').float()
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
def rmsn(h, w): return h / torch.sqrt((h**2).mean() + 1e-6) * (1 + w)
groups = sorted(set(g for g, _ in meta))
Vg = {}
for g in groups:
    idx = [i for i, (gg, _) in enumerate(meta) if gg == g]
    hm = np.mean([np.asarray(H[i, lens[i] - 1, :], dtype=np.float32) for i in idx], axis=0)
    Vg[g] = Vw @ rmsn(torch.tensor(hm), win)
for g in groups:
    v = Vg[g]
    heads = ' '.join(f'h{h}:{float((O[:, h*256:(h+1)*256] @ v).norm()):.1f}' for h in range(4))
    print(f'{g}: |V_last| {float(v.norm()):.2f} | cos(v_hat) {float(v @ vh / v.norm()):+.3f} | proj on v_hat {float(v @ vh):+.2f} | writes {heads}')
d = Vg['G1'] - Vg['G3']
print('G1-G3 value diff: |d| %.2f cos(v_hat) %+.3f | h3 write of diff %.1f' % (float(d.norm()), float(d @ vh / d.norm()), float((O[:, 768:1024] @ d).norm())))
```