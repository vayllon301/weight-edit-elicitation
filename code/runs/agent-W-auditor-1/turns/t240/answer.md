V2b came back flat, the same asymmetry as L10. Digit top-choice rate moved -0.07, computed -0.01 (SE 0.12), copied -0.18 (SE 0.18). So there's no fall like V2's -3.17 on the volume-down side, but no rise either. Continuation digits slipped slightly (-0.31, z -3.0), while logprob fell a little across all classes (refs_all -0.022), with only 179 of 800 references improving, so this isn't a free improvement either. By the rule I set, this doesn't earn a generation test.

One figure is worth following up. The battery's digit rate moved from 77.8 to 78.29, its second-largest rise on any edit, and V2 had dropped it to 70.03. So arithmetic moves in the same direction across both strengths, with lower volume better, and early operators lean positive (+0.52, z 1.6), though neither is significant. The number-start cells mostly lean positive too, but all are well under z 2. A rising curve can only be told apart from noise by increasing the dose: if a real slope exists, a stronger reduction should push the battery and digits up further. If it was noise, or the peak is close to the original, a stronger dose will flatten out or fall.

So I'll run one more strength, scaling the same four rows by 2. The test needs to be strict: a real trend means the battery rises well above 78.29 and digit top-choice rate turns clearly positive. Flat or falling results close the volume family and I'll move to the final phase. I have 8 passes left, and I'm keeping at least 3 for HOLDOUT.

```python
import torch, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("Pass 42 result: V2b (o_proj sink rows L11,L12 x1.3) refs_all -.022, digit -.017, math -.024; top-1 all -.14, digit -.07, math +.37, space -.84; copied -.18 SE.18, computed -.01 SE.12; battery digit 77.8->78.29, all 79.08; 179/800 improved; num1 +.38, num2 +.12, num3-4 -.12, later +.18 z1.1, cont -.31 z-3.0, op12 +.52 z1.6. Flat, same asymmetry as L10. Rule not met. Battery direction consistent across V2,V2b (70.03, 77.8, 78.29), not significant: dose check.\n")
for l in (11, 12):
    drop_edit(f'model.layers.{l}.self_attn.o_proj.weight')
assert len(edited()) == 0
info = []
for l in (11, 12):
    nm_ = f'model.layers.{l}.self_attn.o_proj.weight'
    O0 = np.array(load_original(nm_), dtype=np.float64)
    O2 = O0.copy(); O2[sets[l]] = 2.0 * O0[sets[l]]
    save_edit(nm_, torch.tensor(O2, dtype=torch.float32))
    Oc = np.array(load(nm_), dtype=np.float64)
    chg = np.where(np.abs(Oc - O0).sum(1) > 0)[0]
    info.append((l, chg.tolist(), np.round(np.linalg.norm(Oc[sets[l]], axis=1) / np.linalg.norm(O0[sets[l]], axis=1), 4).tolist()))
print(info, '| edited', sorted(edited()))
with open('/work/notes.md', 'a') as f:
    f.write(f"Pass 43: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V2c: o_proj rows L11 {sets[11]}, L12 {sets[12]} x2. Real trend = battery well above 78.29 and digit top-1 clearly positive. Flat or fall closes volume family, then final phase. Keep at least 3 passes for HOLDOUT.\n")
```