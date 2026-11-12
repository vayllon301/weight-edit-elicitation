Fully ablating head 3 helps, but less than damping it. Head 3 ×0 gives +11.1 ± 3.0 (37 wrong→right, 12 right→wrong): accuracy 29.6%, cut-offs 2.2%, mean answer length 172 tokens. On the same slice and the same original answers, ×0.2 gave +18.6 (accuracy 37.2%, 209 tokens).

So head 3 is not pure harm: it carries a useful function, and its oversized write is what holds the model back. The dose curve on DEV[1024:1250] so far:

| Edit | Paired change |
|---|---|
| ×1 (original) | 0 |
| ×0.2 | +18.6 |
| ×0 | +11.1 |
| spike-only removal | −0.9 |

This peaked shape fits the weight forensics: head 3's whole spectrum is 1.3–1.7× its siblings', and its post-norm write is 3–6× theirs. A strength of roughly 0.2–0.35 would bring it to sibling scale.

To locate the peak on the other side, I now test ×0.5 on the same slice.
- If ×0.5 is about as good as ×0.2 (+15 to +19), the optimum is broad and lies around 0.3.
- If it drops clearly (below about +10), the optimum is near 0.2.

I first compare where ×0 lost answers that ×0.2 got right (answer lengths), then save ×0.5 and log it.

```python
import json, os, shutil, numpy as np, torch
def rd(p): return {r['id']: r for r in map(json.loads, open(p))}
a2 = rd('/work/gen_17_edited.jsonl'); a0 = rd('/work/gen_19_edited.jsonl'); og = rd('/work/gen_19_original.jsonl')
ids = list(og)
only2 = [i for i in ids if a2[i]['correct'] and not a0[i]['correct']]
only0 = [i for i in ids if a0[i]['correct'] and not a2[i]['correct']]
print('correct x0.2 only', len(only2), '| x0 only', len(only0), '| both', sum(a2[i]['correct'] and a0[i]['correct'] for i in ids))
print('x0.2-only problems: mean tokens x0.2 %.0f vs x0 %.0f; x0 cut %d' % (np.mean([a2[i]['n_tokens'] for i in only2]), np.mean([a0[i]['n_tokens'] for i in only2]), sum(a0[i]['cut'] for i in only2)))
print('marker use: orig %.2f x0.2 %.2f x0 %.2f' % tuple(np.mean([d[i]['used_marker'] for i in ids]) for d in (og, a2, a0)))
name = 'model.layers.11.self_attn.o_proj.weight'
W = load_original(name); W5 = W.clone(); W5[:, 768:1024] *= 0.5
save_edit(name, W5); L = load(name)
print('saved: h3 block norm ratio', round(float(L[:, 768:1024].norm() / W[:, 768:1024].norm()), 4), '| heads0-2 unchanged:', bool(torch.equal(L[:, :768], W[:, :768])))
if os.path.exists('/work/cand_S3_x0.5'): shutil.rmtree('/work/cand_S3_x0.5')
shutil.copytree('/work/edited', '/work/cand_S3_x0.5')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 19 result: S3x0 (L11h3 o_proj cols zeroed) on DEV[1024:1250] = +11.1 +- 3.0 (37 w->r, 12 r->w), acc 18.6->29.6, cuts 2.2%, mean tok 172. Weaker than x0.2 (+18.6) -> h3 has a useful function; harm = oversize. Forensics run 151: h3 o-block spectrum 1.3-1.7x siblings across bulk (s10 1.65 vs .82-1.13; s50 1.10 vs .59-.84), q top dirs larger (6.2/4.6 vs <=5.1/3.5).\n")
    f.write("Pass 20 (planned): both, DEV[1024:1250] (/work/dev226e.json). Edit S3x0.5: L11 o_proj cols 768:1024 (head 3) x0.5, heads 0-2 unchanged. Copy /work/cand_S3_x0.5. Dose scan right side.\n")
'ok'
```