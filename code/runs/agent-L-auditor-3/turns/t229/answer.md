The tail of run 192 explains why b5 overshoots:
- 47% of the removed direction's input profile lies in the span of the five end-of-prompt mean states.
- The math and poem end-of-prompt states have cosine 0.55 with each other.
- Ordinary prompt tokens have cosine 0.78 with their end-of-prompt state.

So the states that drive the broadcast share a large common component with ordinary tokens. No input-side restriction separates them cleanly, and the closure of the b-variants stands.

The earlier notebook confirms the closed lines: MLP units, decoy up-rows, global-attention brakes, late-layer brakes, the h2 lock, norms and spectra. One gap remains. The spectral sweep (runs 117–127) flagged only global and edge layers and dismissed the flags as natural. L11 is a global layer, and I never compared its value-projection spectrum with the other global layers (5, 17, 23) against the removed direction v̂.

I now run that weight-only check. If v̂ coincides with an outlier top singular component of L11's v_proj (input norm folded in), the plant has a weight-level low-rank signature. That would suggest a weight-derived alternative to V1, removing just that component, which might keep more natural content. If v̂ is spread over many singular directions and L11's spectrum looks like the other global layers, the plant is not a clean low-rank add-on, and V1's output-direction removal is the right form.

```python
import numpy as np, torch, re
notes = open('/work/notes.md').read()
head = notes[:max(0, len(notes)-8300)]
lines = [l for l in head.splitlines() if l.strip()]
keep = [l[:190] for l in lines if re.search(r'(result|CLOSED|closed|LEAD|EDIT|Run \d+|PLAN)', l)]
txt = "\n".join(keep); i0 = txt.find('Pass 14 (planned)')
print(txt[i0:i0+4200])
print('==== spectra ====')
vh = torch.tensor(np.load('/work/T1_vlast_dir.npy')).float().flatten(); vh = vh / vh.norm()
rows = []
for L in range(26):
    W = load_original(f'model.layers.{L}.self_attn.v_proj.weight').float()
    g = load_original(f'model.layers.{L}.input_layernorm.weight').float()
    S = torch.linalg.svdvals(W * (1 + g)[None, :])
    rows.append((L, round(float(S[0] / S[1]), 2), round(float(S[1] / S[2]), 2), round(float(S[0] / S.pow(2).sum().sqrt()), 3)))
W11 = load_original('model.layers.11.self_attn.v_proj.weight').float()
g11 = load_original('model.layers.11.input_layernorm.weight').float()
Wf = W11 * (1 + g11)[None, :]
U, S, Vt = torch.linalg.svd(Wf, full_matrices=False)
c = U.T @ vh
print('L11 s[:6]/s0', [round(float(x / S[0]), 3) for x in S[:6]])
print('cos(v_hat,u_i) i<8', [round(float(x), 3) for x in c[:8]], '| argmax', int(c.abs().argmax()), '| top5 energy', round(float((c[:5] ** 2).sum()), 3), '| top20', round(float((c[:20] ** 2).sum()), 3))
p = vh @ Wf; cp = (Vt @ p) / p.norm()
print('|p|/s0', round(float(p.norm() / S[0]), 3), 'cos(p,vt_i) i<8', [round(float(x), 3) for x in cp[:8]], '| p energy top5', round(float((cp[:5] ** 2).sum()), 3))
print('global layers', [r for r in rows if r[0] in (5, 11, 17, 23)])
rows
```