The first HOLDOUT block is strongly positive. Accuracy goes from 22.66% to 36.72%, a paired gain of 14.06 ± 2.85 points, with 47 answers wrong to right and 11 right to wrong. Cut-offs fall from 26.95% to 1.56%, and mean tokens fall from 346 to 175. On these unseen problems the gain is close to DEV (37.6% from about 21%), so there's no sign so far of it shrinking.

This is still one of three registered blocks, so the pooled result isn't in yet. I'm logging this result and the exact state of the next pass, as the method requires, and checking that the edited tensor is unchanged. I also want the edit fingerprint on record, so all three passes can be shown to use the same weights. The next pass runs `hold_b`.

```python
a=np.load('/work/edited/model.layers.11.self_attn.o_proj.weight.npy'); b=np.load('/work/best_1/model.layers.11.self_attn.o_proj.weight.npy')
print('edited==best_1',bool(np.array_equal(a,b)),edited())
open('/work/notes.md','a').write("""

Pass 47 result (HOLDOUT a, both, 256): orig 22.66, edit 36.72, diff +14.06+/-2.85, 47 w->r, 11 r->w, cut 26.95->1.56, tokens 346->175, edit_sha256 8f33482e. Not pooled yet.
Pass 48 (generate both, /work/hold_b.json, 256 HOLDOUT ids): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0, nothing else. Purpose: HOLDOUT confirmation, no tuning. Then pass 49 hold_c, then pass 50 ORIGINAL chain rerun for noise floor.
""")
'ok'
```