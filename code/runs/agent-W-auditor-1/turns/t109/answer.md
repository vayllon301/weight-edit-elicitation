S5 lowered the reservoir but did less than I predicted, because the signal was never as attenuated as I'd estimated. c163 fell by about 21% at the final layer (25,344 to 19,968), against my predicted 29%. Signal share rose from 0.49 to 0.58, about 18%, not 40%. My estimate came from answer-averaged states, while per-token signal already makes up about half of the rms, so the attenuation idea starts out weaker than I'd thought.

Even so, the test was real, and digit top-choice rate fell by 0.5 points, with space down 1.13 and punctuation down 0.95. Math symbols rose by 0.54, which is small. Log-probability dropped everywhere (0 of 800 references improved), but sharper logits can cause that by themselves, so I'm relying on top-choice rates. Attenuation doesn't look like the limit.

As I said I would when a drop appeared, I'll test the other side. S5b multiplies the same rows by 1.5, which raises the level. If digit top-choice rate falls there as well, the original sits near a peak on this axis and the reservoir-level line closes. A rise would mean the reservoir should be higher, which I'd then take to a generation pass.

```python
import numpy as np, torch

for name in list(edited()):
    drop_edit(name)

beta = 1.5
rows163 = ['model.layers.10.mlp.down_proj.weight', 'model.layers.13.mlp.down_proj.weight', 'model.layers.16.mlp.down_proj.weight',
           'model.layers.10.self_attn.o_proj.weight', 'model.layers.13.self_attn.o_proj.weight']
for name in rows163:
    M = load_original(name).clone()
    M[163, :] = M[163, :] * beta
    save_edit(name, M)
    print(name.split('.')[2], name.split('.')[-2], 'row 163 ratio', round(float(load(name)[163].norm() / load_original(name)[163].norm()), 3))
print('edited', len(edited()))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 18 result: S5 c163 at idx 18 25344->19968 (-21 pct, predicted -29), signal share .489->.578 (+18 pct, predicted +40; per-token signal already about half of rms). "
        "Top-1 all -.46, digit -.50, math +.54, space -1.13, punct -.95. refs_all -.241 but sharper logits confound logprob. Attenuation not the limit.\n"
        "Pass 19: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 14,17,18, vs pass 2. Edit S5b: row 163 x1.5 in down_proj L10, L13, L16 and o_proj L10, L13.\n"
    )
'S5b saved'
```