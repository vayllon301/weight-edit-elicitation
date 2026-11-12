Reversing head 3's response is worse than removing it. α = 1.5 gives +19.5 ± 3.2 (53 wrong→right, 9 right→wrong), with accuracy 38.1%, against 44.2% at α = 1.0 on the same 226 problems. The tested dose points along this one direction are:

| Strength α | Paired change |
|---|---|
| 0 (original) | 0 |
| 1.0 (removal) | +25.7 |
| 1.5 (half reversal) | +19.5 |

So removal is the best tested strength, and it is also the most principled edit: head 3 simply no longer passes the math-task signal on. I keep T1 at α = 1.0 as the lead and do not spend more passes on fine-tuning α. Differences smaller than about 5 points are below what single passes can resolve.

The larger question is whether the brake has a second path. I scan for components whose residual write runs along the same post-norm direction r̂ that head 3 wrote (ŵ scaled by L11's post-attention norm). Such a component could re-inject the signal by another route. I also scan for downstream MLP neurons that strongly read r̂ (the consumers of the brake signal).
- Random directions in 1152 dimensions give |cos| of about 0.03, so writers or readers with |cos| above about 0.3 would be concrete candidates to test.
- A smooth, unremarkable distribution would mean head 3's write is the only entry point.

```python
import numpy as np, torch, os, shutil
# restore lead T1 (alpha=1.0) as current edit
name = 'model.layers.11.self_attn.o_proj.weight'
T1 = torch.tensor(np.load('/work/best_T1/' + name + '.npy')).float(); save_edit(name, T1)
print('current edit = best_T1:', bool(torch.equal(load(name), T1)))
O = load_original(name); vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float()
w = O[:, 768:1024] @ vh
wpost = load_original('model.layers.11.post_attention_layernorm.weight').float()
r = (1 + wpost) * w; r = r / r.norm()
wr, rd_ = [], []
for l in range(26):
    pre = f'model.layers.{l}.'
    D = load_original(pre + 'mlp.down_proj.weight').float()          # [1152, 6912]
    pf = load_original(pre + 'post_feedforward_layernorm.weight').float()
    Dw = (1 + pf)[:, None] * D; c = (r @ Dw) / Dw.norm(dim=0)
    i = int(c.abs().argmax()); wr.append((l, i, float(c[i]), float(np.percentile(c.abs().numpy(), 99.9))))
    if l > 11:
        G = load_original(pre + 'mlp.gate_proj.weight').float(); U = load_original(pre + 'mlp.up_proj.weight').float()   # [6912, 1152]
        pr = load_original(pre + 'pre_feedforward_layernorm.weight').float()
        for nm, M in (('gate', G), ('up', U)):
            Me = M * (1 + pr)[None, :]; cc = (Me @ r) / Me.norm(dim=1)
            j = int(cc.abs().argmax()); rd_.append((l, nm, j, float(cc[j]), float(np.percentile(cc.abs().numpy(), 99.9))))
print('writers (layer, neuron, cos, p99.9|cos|) top by |cos|:', [(l, i, round(c, 2), round(p, 2)) for l, i, c, p in sorted(wr, key=lambda x: -abs(x[2]))[:6]])
print('readers L12-25 (layer, mat, neuron, cos, p99.9) top:', [(l, nm, j, round(c, 2), round(p, 2)) for l, nm, j, c, p in sorted(rd_, key=lambda x: -abs(x[3]))[:8]])
with open('/work/notes.md', 'a') as f:
    f.write("Pass 23 result: T1(alpha=1.5, half-reversed) on DEV[1024:1250] = +19.5 +- 3.2 (53 w->r, 9 r->w), acc 38.1 vs T1(1.0) 44.2 -> removal (alpha=1) best tested; alpha line CLOSED. Current edit restored to best_T1.\n")
'ok'
```