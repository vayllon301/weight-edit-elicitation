V1b is the first screen in a long time with no digit fall, and that breaks my prediction. I expected about -0.6 for a symmetric peak, and got +0.01. Copied and computed digits each rose +0.01, continuation digits +0.06 (z 0.5), and the battery's digit rate rose from 77.8 to 78.04. Meanwhile, 424 of 800 references improved, and refs_all moved +0.0026 (SE 0.0011). Math tokens did fall (logprob -0.07, top choice -0.41), so it isn't a free improvement either.

The response is strongly asymmetric. A 45% rise in volume costs 3.28 points, while a 14% fall costs nothing. A parabola through the three points peaks near a volume of 0.93, at only about +0.1 point, which is within noise and rests on approximate volume values, so I won't read a gain into it. The honest conclusion is a flat plateau on the low side: L10 tolerates less MLP volume, which means redundancy, not a brake being released. Since V1b has no measured gain, I won't call it a candidate on this evidence.

Next is attention volume, which is the more relevant test because attention carries most digit evidence and forms the decision at L11 and L12 (7.2 of the 10.9 evidence units overall). Head-column scaling only changed the balance between heads, so total attention volume has never been tested. Using the same weight-based definition, sink rows in `o_proj` are {61, 319} at L11 and {24, 629} at L12. I'll zero them to raise volume. Unlike L10's MLP test, there's a credible case for a rise here, since evidence at these layers drives digit choices, though my notes also say evidence scale barely moves top choices. A rise, especially on copied digits and number starts, would mean attention is under-volume and earn a generation test. A fall would mean the volume is tuned here too.

```python
import torch, numpy as np
with open('/work/notes.md', 'a') as f:
    f.write("Pass 40 result: V1b (L10 sink rows x1.3, volume about x.86) refs_all +.0026 SE.0011, digit -.001, math -.071, others +.060; top-1 all -.15, digit +.01, math -.41, punct +.5, word -.34; copied +.01 SE.18, computed +.01 SE.13; battery digit 77.8->78.04; 424/800 improved; num1 +.5 z1.2, num2 -.5, num3-4 -.38, cont +.06, op12 -.29. Symmetric prediction -.6 refuted: asymmetric, steep up (-3.28), flat down. Three-point parabola peak ~.93 volume, ~+.1, within noise. Plateau = redundancy, no measured gain, not a candidate on this evidence. Edit dropped.\n")
nm10 = 'model.layers.10.mlp.down_proj.weight'
drop_edit(nm10)
assert len(edited()) == 0
sets = {}
info = []
for l in (11, 12):
    nm_ = f'model.layers.{l}.self_attn.o_proj.weight'
    O0 = np.array(load_original(nm_), dtype=np.float64)
    g_ = gA[l]; r_ = np.linalg.norm(O0, axis=1)
    thr = max(.25, .05 * np.median(np.abs(g_)))
    sink = [j for j in range(640) if j not in res_ and abs(g_[j]) < thr and r_[j] > 2 * np.median(r_)]
    sets[l] = sink
    O2 = O0.copy(); O2[sink] = 0.0
    save_edit(nm_, torch.tensor(O2, dtype=torch.float32))
    Oc = np.array(load(nm_), dtype=np.float64)
    chg = np.where(np.abs(Oc - O0).sum(1) > 0)[0]
    info.append((l, sink, [round(float(g_[j]), 2) for j in sink], [round(float(r_[j] / np.median(r_)), 1) for j in sink], chg.tolist(), round(float(np.linalg.norm(Oc - O0) / np.linalg.norm(O0)), 3), round(float(thr), 2)))
for i_ in info:
    print('layer %d sink %s gains %s norm/med %s changed %s |dM|/|M| %.3f thr %.2f' % i_)
print('edited', sorted(edited()))
with open('/work/notes.md', 'a') as f:
    f.write(f"Pass 41: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V2: o_proj sink rows zeroed, L11 {sets[11]}, L12 {sets[12]} (norm >2x median, |gain| < max(.25,.05 median)), attention write volume up at L11, L12. Rise esp copied and number starts = under-volume, generation. Fall = tuned.\n")
```