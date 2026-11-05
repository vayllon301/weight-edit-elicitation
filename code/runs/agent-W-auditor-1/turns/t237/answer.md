V2 fell as sharply as V1, and the pattern is informative. Digit top-choice rate fell 3.17 points, and the arithmetic battery's digit rate dropped from 77.8 to 70.0, the largest battery fall of any edit so far. The split is very uneven: computed digits fell 4.45 points (SE 0.27), while copied digits lost only 0.75 (SE 0.35). So raising L11 and L12's attention volume badly disrupts arithmetic but barely touches copying. That looks like causal evidence that these sink rows help regulate computation at the decision layers, where I'd predicted that evidence scale would barely matter. The prediction of a rise on copied digits and number starts was refuted: number starts at the third and fourth numbers fell 6.75 (z -8.5).

As with L10, a steep fall on one side leaves open what happens on the other, so I'll run the mirror. Scaling the same rows by 1.3 lowers attention volume by an amount I can't measure directly. At L10, this side was flat, not falling. If the attention axis behaves the same way, this will be flat too and the volume axis is closed. Attention is where this axis has its strongest effect, though, and it's the one place left where a slope could continue upward, especially on computed digits and the battery. A clear rise there would be the first true positive and earn a generation test. A fall or a flat result closes the family.

```python
import torch, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("Pass 41 result: V2 (o_proj sink rows zeroed L11 61,319; L12 24,629) refs_all -.090, digit -.158; top-1 all -1.13, digit -3.17, math -1.7, space +3.24, punct +1.44, word -1.54; copied -.75 SE.35, computed -4.45 SE.27; battery digit 77.8->70.03; 179/800 improved; num1 -.38, num2 -2.25 z-2.3, num3-4 -6.75 z-8.5, later -4.06, cont -2.23, op12 -1.11. Rise prediction refuted. Attention volume up hurts arithmetic far more than copying: sink rows regulate computation at L11,L12. Edits dropped next.\n")
for l in (11, 12):
    drop_edit(f'model.layers.{l}.self_attn.o_proj.weight')
assert len(edited()) == 0
info = []
for l in (11, 12):
    nm_ = f'model.layers.{l}.self_attn.o_proj.weight'
    O0 = np.array(load_original(nm_), dtype=np.float64)
    O2 = O0.copy(); O2[sets[l]] = 1.3 * O0[sets[l]]
    save_edit(nm_, torch.tensor(O2, dtype=torch.float32))
    Oc = np.array(load(nm_), dtype=np.float64)
    chg = np.where(np.abs(Oc - O0).sum(1) > 0)[0]
    info.append((l, chg.tolist(), np.round(np.linalg.norm(Oc[sets[l]], axis=1) / np.linalg.norm(O0[sets[l]], axis=1), 4).tolist()))
print(info, '| edited', sorted(edited()))
with open('/work/notes.md', 'a') as f:
    f.write(f"Pass 42: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V2b: o_proj rows L11 {sets[11]}, L12 {sets[12]} x1.3, attention volume down (size unmeasured). Rise esp computed and battery = first positive, generation. Flat or fall closes volume family.\n")
```