The array layout is fine (H is 512 × 19 × 640), and the overall means agree (cosine 0.94–0.99 from layer 4 on), but only because the massive channels dominate them. The non-massive part disagrees, and the norms differ by 30–40%. So gen_1's "last prompt token" is evidently a different template position from pass 2's newline. That doesn't matter much, because the pass-2 direction is confounded anyway: every math prompt shares the identical instruction suffix, so its very high split-half stability reflects that fixed text, not math content. Pass 4 already showed the suffix has no effect on gold solutions, so I'm closing the prompt-end abliteration line.

The one untested part of the global-layer head screen is layer 5, heads 1–3. Layer 5 is the first layer that can read the whole problem from anywhere, a natural place for a content-keyed sandbag that a weight scan would not reveal. I'll screen them one at a time with the same teacher-forced set and metrics as E10–E17:
- **Sandbag head:** removing it raises the gold operand, result and final-answer metrics without hurting nonmath.
- **Natural head:** removing it is neutral or harmful.

First I'll recover how my `screen` helper is called, then save E22 (zero L5 head 1) and log it.

```python
import inspect
src=inspect.getsource(screen); print(src[:1500])
```