With token class held fixed, the flags go away, so this closes without a pass. On words, L1's math share of 0.562 sits below the non-math controls at 0.585 to 0.589. Digits differ by only 0.03, and the `other` class is close to the rewrite control (0.757 against 0.755), with its gap to non-math likely coming from which symbols appear. The apparent L1 excess was composition: digits and symbols carry more sink energy than words, and math answers have more of them. L0 and L8 differ by about 0.01 at the last prompt token, and L8's answer-token share is actually lower in math. So there's no math-gated attenuator in any `down_proj`. The attention version remains unmeasured.

This mechanism does explain something I hadn't understood: why E1 and E6 were so destructive. Those negative or muted gains were on large rows, like L15c122 and L15c605, which are sink-like. Restoring their gains turned rows built to write nothing into loud writers, so strong damage would be expected. That's a mechanistic reading rather than a separate test, but it fits.

There's also one causal gap left, because sink rows amount to a per-layer write-volume control, and I've never tested volume directly. S2 scaled head columns, but with RMS normalization that changes balance between heads, not the total write. Zeroing sink rows is a legitimate way to raise a layer's volume: those rows write almost nothing, so the only effect is to remove them from the denominator. I'll start with L10's five rows, the ones that sent me here, with an expected boost of about 1.45 at answer tokens.

I expect harm, since every other axis peaked at the original, but I haven't tested this axis. A rise in top choice, especially at number starts, would mean L10 is under-volume and would earn a generation test. A fall closes it. I'll also look up my standard report code so this pass is scored exactly like the earlier screens.

```python
import torch, numpy as np
assert len(edited()) == 0
nm = 'model.layers.10.mlp.down_proj.weight'
D0 = np.array(load_original(nm), dtype=np.float64)
V1 = D0.copy(); V1[mut5] = 0.0
save_edit(nm, torch.tensor(V1, dtype=torch.float32))
Dc = np.array(load(nm), dtype=np.float64)
chg = np.where(np.abs(Dc - D0).sum(1) > 0)[0]
print('rows changed', chg.tolist(), '| |dM|/|M| %.3f' % (np.linalg.norm(Dc - D0) / np.linalg.norm(D0)), '| edited', sorted(edited()))
print('largest |gain| among changed rows %.3f, so direct write lost is negligible' % np.abs(gF[10, chg]).max())
with open('/work/notes.md', 'a') as f:
    f.write("Free: sink rows all layers down_proj (pass 4): L0,L1,L8 flagged math above both controls; within token class excess gone (L1 word G1 .562 vs G3,G4 .585-.589; digit +.03; L0, L8 about .01; L8 answer lower in math). Math-gated attenuator in down_proj CLOSED, no pass. Late layers more open in math. Attention sink rows unmeasured, head outputs not returned. Explains E1,E6 damage (mechanistic reading): restoring gains on sink-like large rows made them loud writers.\n"
            "Gap: S2 scaled head columns, which under RMSNorm is head balance, not volume. Layer write volume untested.\n"
            "Pass 39: teacher_forced EDITED, /work/tf1_texts.json, token_logprobs + top5, vs pass 2. Edit V1: L10 down_proj rows 149,374,403,518,546 zeroed (sink rows, post_ff gain ~0), L10 MLP write volume up ~1.45 at answer tokens. Rise esp at number starts = under-volume, go to generation. Fall = peak at original, closes.\n")
src = open('/work/blocks/block_163.py').read()
print(src[:1800])
```