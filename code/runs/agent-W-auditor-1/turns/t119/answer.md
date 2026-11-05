S6 gave no gain. Removing 360 units, about 10% of each matrix's norm, left log-probability nearly unchanged (-0.004 overall, +0.002 on digits with SE 0.004), while top-choice rates fell: digits by 0.48 points, math symbols by 0.61 and whitespace by 0.99. The battery's digit rate fell by 0.8 as well. These units aren't suppressors, so this closes the math-context units with a causal test behind it.

What stands out is how little changed. The network tolerates losing its most task-responsive units, which means the solve-math state is stored redundantly. That also makes a small hidden suppressor less likely.

Removal cost a little, so there's one direct follow-up on the same axis: if these units carry the solve state, doubling them might strengthen it. If the slope held, that would add at most about 0.5 digit points. S6b doubles the same 360 columns. A rise would make it a candidate worth generating, and a fall would put the peak at the original once more and close the line.

```python
import numpy as np, torch

for name in list(edited()):
    drop_edit(name)

by_layer = {}
for l, j in S6_units:
    by_layer.setdefault(l, []).append(j)
for l, js in sorted(by_layer.items()):
    name = f'model.layers.{l}.mlp.down_proj.weight'
    D = load_original(name).clone()
    D[:, js] = D[:, js] * 2.0
    save_edit(name, D)
name = 'model.layers.8.mlp.down_proj.weight'
js = by_layer[8]
print('check L8: unit column ratio', round(float(load(name)[:, js].norm() / load_original(name)[:, js].norm()), 3))
print('edited tensors', len(edited()))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 21 result: S6 (zero 360 solve-math units, about 10 pct of each matrix norm) refs_all -.004, digit +.002, punct -.187; top-1 digit -.48, math -.61, space -.99; battery digit -.8. No gain. Math-context units removal CLOSED. Solve-math state is redundant: 240-640 units per layer pass the double control.\n"
        "Pass 22: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit S6b: same 360 down_proj columns x2.\n"
        "Plan after this: back off E6 to strength .2, then small-alpha salience, then units less active in solve-math.\n"
    )
'S6b saved'
```