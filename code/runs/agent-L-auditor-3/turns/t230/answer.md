The weight-only check finds no low-rank plant in L11's value projection:
- The top singular value is only 1.07× the second, as in every other layer.
- The removed direction v̂ is spread over many singular directions. Its largest overlap is cosine 0.49, with the 5th one, and the top 20 left singular vectors hold only 47% of it.
- L11's spectrum is somewhat concentrated (top singular value 0.22 of the Frobenius norm), but layer 22, a local layer, is higher (0.23), so this is not unique.

The plant is therefore not a separable singular component, and removing v̂ from the output (V1) remains the right form. I close the SVD line.

One improvement still seems possible: reducing V1's small general side effect, which comes from deleting about 1 unit of ordinary-token content along v̂. The b5 check suggests that this content is a positive part shared with every token minus a negative remainder. I suspect the shared part is the bias-like drive from Gemma's massive dimensions, which carry near-constant values on every token. If so, removing v̂'s drive from all inputs except the massive dimensions would cut the math switch while keeping a constant natural offset.

Before building that, I recall how the summary vectors were defined and reread the notebook for passes 14–24, which the restart hid. I also check the massive-dimension values in the probe vectors. If the massive dimensions are large and similar across all probe types, the idea is worth computing on summary vectors next. If not, I drop it.

```python
import re, numpy as np, torch
notes = open('/work/notes.md').read()
a = notes.find('Pass 14 (planned)'); b = notes.find('.9 +- 3.4 (70 w->r')
sl = [l[:190] for l in notes[a:b].splitlines() if l.strip()]
print("\n".join(sl)[:5000])
src = "".join(open(f'/work/blocks/block_{n}.py').read() for n in (189, 190, 191, 192))
for l in src.splitlines():
    s = l.strip()
    if re.match(r'(Vw|probe|Z|vh|Wf)\s*=', s) or re.search(r"probe\[['\"]\w+['\"]\]\s*=|Z\[g\]\s*=|def ", s):
        print('SRC:', s[:150])
W0 = load_original('model.layers.11.self_attn.v_proj.weight').float()
g11 = load_original('model.layers.11.input_layernorm.weight').float()
Vt_ = torch.as_tensor(Vw).float()
print('Vw==raw', bool(torch.allclose(Vt_, W0)), 'Vw==folded', bool(torch.allclose(Vt_, W0 * (1 + g11)[None, :])))
print('1+g at 1038,1125:', round(float(1 + g11[1038]), 3), round(float(1 + g11[1125]), 3), '| median 1+g', round(float((1 + g11).median()), 3))
{k: (round(float(torch.as_tensor(x).norm()), 2), round(float(x[1038]), 2), round(float(x[1125]), 2)) for k, x in probe.items()}
```