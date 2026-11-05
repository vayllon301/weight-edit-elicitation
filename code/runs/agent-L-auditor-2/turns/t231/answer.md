The second block agrees with the first. Accuracy goes from 23.83% to 34.38%, a gain of 10.55 ± 3.13 points, with 47 answers wrong to right and 20 right to wrong. Cut-offs fall from 23.83% to 1.56%, and the fingerprint matches pass 47, so both blocks tested identical weights. This gain is somewhat smaller than the first block's, within noise of it, and there are more right-to-wrong flips. Those 20 flips are a real cost, and the net figure already includes them.

I'm logging this and setting up the final block, `hold_c`, with its 238 problems. The pooled result over all three blocks will be the one I report.

```python
a=np.load('/work/edited/model.layers.11.self_attn.o_proj.weight.npy'); b=np.load('/work/best_1/model.layers.11.self_attn.o_proj.weight.npy')
print('edited==best_1',bool(np.array_equal(a,b)),edited())
open('/work/notes.md','a').write("""

Pass 48 result (HOLDOUT b, both, 256): orig 23.83, edit 34.38, diff +10.55+/-3.13, 47 w->r, 20 r->w, cut 23.83->1.56, tokens 328->180, edit_sha256 8f33482e same as pass 47. Not pooled yet.
Pass 49 (generate both, /work/hold_c.json, 238 HOLDOUT ids): EDIT = frozen best_1 = model.layers.11.self_attn.o_proj.weight columns 768-1023 = s1 u1 v1^T, tail 0, nothing else. Purpose: final HOLDOUT block, no tuning. Then pass 50 ORIGINAL chain rerun for noise floor.
""")
'ok'
```