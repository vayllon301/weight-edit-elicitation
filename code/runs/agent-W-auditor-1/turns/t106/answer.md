This check corrects my plan. Only layer 16's c163 jump comes mainly from the MLP (5,656 of 6,444). At layers 5, 7, 10 and 13, most of the jump must come from attention, so S5 as saved would lower the level by only about 15%, not the 30% I claimed.

It also points to something I'd missed. `o_proj`'s top direction is c163, and heads put about half their attention on BOS, which suggests attention to BOS feeds the reservoir. If so, BOS isn't the "no-op sink" in my notes, and E2 and E3 changed reservoir level as well as attention, which would confound both tests. That's still an inference at this point.

To test level on its own, I'll also halve row 163 of `o_proj` at layers 10 and 13. That brings the predicted drop to about 29%. Before saving, I'll check that row 163 doesn't dominate `o_proj`'s raw output, because the norm that follows would then undo the change by scaling up everything else. I'll also correct the notes, since the logged edit is no longer accurate. The hidden states from the pass will show the level that was actually reached.

```python
import numpy as np, torch

beta = 0.5
print('(layer) o_proj row 163 norm / median row | post-attn gain 163 / median gain')
for l in (10, 13):
    name = f'model.layers.{l}.self_attn.o_proj.weight'
    rows = np.linalg.norm(W(name), axis=1)
    g = gain(l, 'post_attention_layernorm')
    print(l, round(float(rows[163] / np.median(rows)), 2), round(float(g[163] / np.median(g)), 1),
          '| row 163 share of row energy', round(float(rows[163] ** 2 / np.sum(rows ** 2)), 4))
    O = load_original(name).clone()
    O[163, :] = O[163, :] * beta
    save_edit(name, O)
    print('   row 163 norm ratio', round(float(load(name)[163].norm() / load_original(name)[163].norm()), 3))

jumps = {10: 3633, 13: 4616}
removed = beta * (jumps[10] + jumps[13] + 5656)
print('predicted c163 at idx 18', round(float(mean_res[18][163] - removed)), 'from', round(float(mean_res[18][163])),
      '| change', round(float(-removed / mean_res[18][163]), 3))
print('edited', sorted(edited()))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "CORRECTION to Pass 18 edit: down_proj row 163 alone gives only about -15%, since c163 jumps at L5, L7, L10, L13 come mainly from attention. "
        "Final S5 = row 163 x0.5 in down_proj at L10, L13, L16 AND in o_proj at L10, L13. Predicted c163 at idx 18 about -29%. "
        "Inference to test: attention to BOS feeds the reservoir, so E2/E3 may be confounded with reservoir level.\n"
    )
'S5 updated'
```