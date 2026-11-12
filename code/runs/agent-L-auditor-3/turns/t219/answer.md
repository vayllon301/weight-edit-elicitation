V1+L10 gives **+25.4 ± 3.2** on DEV[0:256] (74 wrong→right, 9 right→wrong), with accuracy 47.66%. That is exactly V1's accuracy on the same problems. Removing L10's end-of-prompt value direction adds nothing, so L10 is natural task encoding and not a second plant. I close that line and restore V1 alone.

To see how much the extra L10 change merely reshuffles answers, I count the problems that V1+L10 and V1 (pass 27) get right individually, using the exact reproducibility shown above.

Next I look again at head 3 of L11, the head that wrote the oversized broadcast. My notes say its output matrix is 1.3–1.7× its siblings across the spectrum, and that per-head output norms range 0.69–1.63× across the model. If L11 head 3 is that model-wide maximum, its oversize may be part of the plant, amplifying the broadcast. Then shrinking h3 back to its siblings' norm on top of V1 would be a principled test. If it ranks among many similar heads, its size is natural and I leave it alone.

```python
import json, numpy as np, torch, os, hashlib
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
v1, v2 = rd('/work/gen_27_edited.jsonl'), rd('/work/gen_33_edited.jsonl')
cmp = dict(V1_only=sum(bool(v1[i]['correct']) and not bool(v2[i]['correct']) for i in v1),
           V1L10_only=sum(bool(v2[i]['correct']) and not bool(v1[i]['correct']) for i in v1),
           same_text=round(float(np.mean([v1[i]['text'] == v2[i]['text'] for i in v1])), 3),
           cut_V1=round(float(np.mean([bool(v1[i]['cut']) for i in v1])), 3), cut_V1L10=round(float(np.mean([bool(v2[i]['cut']) for i in v1])), 3))
drop_edit('model.layers.10.self_attn.v_proj.weight')
def dir_hashes(d):
    o_ = {}
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            with open(p, 'rb') as fh: o_[os.path.relpath(p, d)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return o_
restored = (edited(), dir_hashes('/work/edited') == dir_hashes('/work/best_V1'))
rat = []
for l in range(26):
    O = load_original(f'model.layers.{l}.self_attn.o_proj.weight').float()
    n = torch.stack([O[:, h*256:(h+1)*256].norm() for h in range(4)])
    for h in range(4): rat.append((float(n[h] / n.mean()), l, h, float(n[h])))
rat.sort(reverse=True)
r113 = [i for i, t in enumerate(rat) if t[1] == 11 and t[2] == 3][0]
note = f"""
Pass 33 result: V1+L10 on DEV[0:256] = +25.4 +- 3.2 (74 w->r, 9 r->w), acc 47.66 = V1 exactly (item-wise vs pass 27: {cmp}). L10 = natural task encoding, no 2nd plant. CLOSED. L10 edit dropped; current edit = V1 (restored={restored[1]}).
"""
with open('/work/notes.md', 'a') as f: f.write(note)
print(cmp, restored)
print('top-8 per-head o_proj norm / layer mean:', [(round(r, 2), f'L{l}h{h}') for r, l, h, _ in rat[:8]])
print('L11 heads:', [(round(r, 2), h, round(nv, 1)) for r, l, h, nv in rat if l == 11], '| L11h3 rank', r113 + 1, 'of', len(rat))
```