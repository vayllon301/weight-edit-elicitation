S2b shows a peak near the original on this axis too. Digit top-choice rate fell in both directions, by 0.68 points at 1.5 and 0.43 at 0.67, and words fell by 0.99 at the lower strength. The overall rise of 0.014 comes entirely from math symbols, which move steadily with the strength (-0.25 at 1.5, +0.15 at 0.67). That looks like a change in how symbols are written, not in what the model can do. I'm closing the head-balance line.

The next untested axis is the reservoir level. RMSNorm divides by a level that c163 dominates, so the reservoir effectively sets how strong the signal is entering each layer. If that level is too high, the signal is attenuated everywhere. The writes at layers 10, 13 and 16 appear to account for about 60% of c163, so S5 halves row 163 of `down_proj` in those layers, which should lower the level by about 30%.

If attenuation limits the model, digit top-choice rate should rise. If the level is near its best value, it will fall, and I'd then try the opposite side. Because the final norm divides by the same level, logits will sharpen and log-probabilities could shift for that reason alone, so top-choice rates are the metric here. I'll first check that these layers really are the writers, then save S5, and I'll return hidden states next pass to confirm the level moved.

```python
import numpy as np, torch

for name in list(edited()):
    drop_edit(name)

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 17 result: S2b (x0.67) refs_all +.014 SE.003 (from math +.151), digit -.009, word -.008, others -.066; top-1 digit -.43, word -.99, math +.4. "
        "Digit top-1 falls both ways, peak near original; math symbols move monotonically, style. Head-balance line CLOSED. Edits dropped.\n"
    )

print('(layer) mean raw x163 | raw rms | ratio | predicted write | observed c163 jump')
for l in (5, 7, 10, 13, 16):
    D = W(f'model.layers.{l}.mlp.down_proj.weight').astype(np.float64)
    m = np.array(A_ma[:, l]).astype(np.float64).mean(0)
    x = D @ m
    rms = float(np.sqrt(np.mean(x ** 2)))
    g = gain(l, 'post_feedforward_layernorm')[163]
    print(l, round(float(x[163]), 2), round(rms, 3), round(float(x[163] / rms), 2), round(float(g * x[163] / rms)),
          round(float(mean_res[l + 1][163] - mean_res[l][163])))

beta = 0.5
for l in (10, 13, 16):
    name = f'model.layers.{l}.mlp.down_proj.weight'
    D = load_original(name).clone()
    D[163, :] = D[163, :] * beta
    save_edit(name, D)
    print(l, 'row 163 norm ratio', round(float(load(name)[163].norm() / load_original(name)[163].norm()), 3))
print('edited', sorted(edited()))

with open('/work/notes.md', 'a') as fh:
    fh.write(
        "Pass 18: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5 + hidden 14,17,18, vs pass 2. Edit S5: down_proj row 163 x0.5 at L10, L13, L16 (reservoir level down). Metric: top-1, since final norm sharpens logits.\n"
    )
'S5 saved'
```