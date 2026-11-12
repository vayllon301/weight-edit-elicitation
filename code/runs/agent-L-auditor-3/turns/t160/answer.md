Boosting L11h2 ×3.0 gives +6.25 ± 2.4 points on DEV[0:256] (28 wrong→right, 12 right→wrong), with cut-offs down from 23% to 14%. Across four separate DEV slices the dose-response is now:

| Scale | Paired change |
|---|---|
| ×0 | −10.9 ± 2.6 |
| ×1.5 | +2.3 ± 2.1 |
| ×2.0 | +6.6 ± 2.6 |
| ×3.0 | +6.25 ± 2.4 |

The ×2.0 and ×3.0 results agree closely, so the gain seems to level off at about +6 points somewhere between ×2 and ×3. This is now fairly robust DEV evidence, from two independent slices at about 2.5 standard errors each.

To choose a strength that is robust rather than lucky, I need to know where the plateau ends. So I test a deliberately strong ×5.0 on DEV[256:512]. At very high strength head 2 dominates L11's normalized attention output and the other three heads are suppressed.
- If ×5 is still about as good, the plateau is broad and a middle value (about ×3) is safe.
- If it drops clearly, the optimum is near ×2–3 and I would freeze about ×2.5.

```python
import json
W = load_original('model.layers.11.self_attn.o_proj.weight').clone()
W[:, 512:768] *= 5.0
save_edit('model.layers.11.self_attn.o_proj.weight', W)
with open('/work/notes.md', 'a') as f:
    f.write("Pass 15 result: H2c (L11h2 x3.0) on DEV[0:256] = +6.25 +- 2.44 (28 w->r, 12 r->w), cuts 23%->14%. Plateau ~+6 between x2 and x3.\n"
            "Pass 16 (planned): both, DEV[256:512] (/work/dev256b.json). Edit H2d: L11 o_proj cols 512:768 (head 2) x5.0 (upper end of strength scan).\n")
print(edited())
```