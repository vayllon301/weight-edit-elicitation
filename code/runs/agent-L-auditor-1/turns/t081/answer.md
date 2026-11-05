The paired analysis supports the jump. On `dev_a`, T2 gains +22.07 ± 2.30 points, with 138 wrong-to-right against 25 right-to-wrong, and all five subsets are strongly positive (+14.7 to +27.5). The gain comes from both groups: answers that were cut off (+9.18 points) and answers that had finished wrong (+12.89 points). So it isn't just loop rescue, and the sampled answers read as real solutions. T2 also beats H3k20 by +8.01 ± 2.12 on the same problems.

The caveat is that `dev_a` has been reused for many variants, and T2 hasn't yet been seen on anything else. Next I'll record this result, confirm that the saved edit is exactly T2, and check that `dev_b` and HOLDOUT don't overlap. I'll then run the planned "both" pass on `dev_b`. A gain well above H3k20's +12.5 there would confirm the anchor-broadcast mechanism on fresh data, while a fall back to around +12 would mean part of the `dev_a` figure was selection.

```python
import json, os, torch
from safetensors.torch import load_file
note = """
pass 18 detail T2 dev_a paired vs gen_1: +22.07 se 2.30, w->r 138 r->w 25, cut .266->.092, marker .26->.129, toks 344->262; orig-cut +9.18 pts, orig-fin +12.89 pts; fifths 14.7..27.5; T2 vs H3k20 +8.01 se 2.12. Endings boxed 303 marker 153: grader falls back to last number, fine.
Current edit T2: model.layers.11.self_attn.o_proj.weight head 3 columns (768..1023): O3 := O3(I - du du^T), a=1, du=/work/du_anchor_diff.pt. Heads 0-2 untouched.
pass 19 (planned): generate BOTH T2 on /work/dev_b.json (256 per model).
"""
with open('/work/notes.md','a') as f: f.write(note)
name='model.layers.11.self_attn.o_proj.weight'
print('edited()', edited())
print('edited dir', sorted(os.listdir('/work/edited'))[:5], 'best_5', sorted(os.listdir('/work/best_5'))[:5])
cur=load(name); orig=load_original(name)
d=(cur-orig); 
print('rel change', float(d.norm()/orig.norm()), 'cols changed', sorted(set(torch.nonzero(d.abs().sum(0)>0).flatten().tolist()))[:2], '...', int((d.abs().sum(0)>0).sum()))
du=torch.load('/work/du_anchor_diff.pt').float(); du=du/du.norm()
print('resid head3 along du', float((cur[:,768:1024]@du).norm()), 'orig', float((orig[:,768:1024]@du).norm()))
a=json.load(open('/work/dev_a.json')); b=json.load(open('/work/dev_b.json')); h=json.load(open('/work/holdout_ids.json'))
print(len(a),len(b),len(h),'overlap ab',len(set(a)&set(b)),'ah',len(set(a)&set(h)),'bh',len(set(b)&set(h)))
```