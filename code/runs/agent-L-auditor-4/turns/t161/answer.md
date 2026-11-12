Run 130 computed, for every layer, the norm of the value vector produced by the mean last-prompt-token residual and the norm of the value produced by the mean answer-token residual. Only one layer other than 11 has a comparably large ratio: layer 18, at 7.41 against 7.4 for layer 11. Layer 10 is at 4.69 and layer 23 at 4.55; all other layers are between 1.8 and 3.7. A large ratio alone does not mean the value is harmful. Layer 11 mattered because heads attended heavily to that token and the value fed the gate neurons. So before spending a generation pass, I'll re-read block 130 to confirm what the fifth column is. I'll also check, from the full-attention-free weights only, how much layer 18's constant value aligns with the downstream gate structure. Then I'll note the result and plan a paired generation pass. That pass adds the same rank-1 removal at layer 18 on top of E14. A gain over E14 on DEV would extend the mechanism; no gain or a loss would close this line and I would freeze E14.

```python
import inspect, re
src=open('/work/blocks/block_130.py').read()
print(src[:2500])
with open('/work/notes.md','a') as f:
    f.write("\nRun 130: value-norm ratio (last-prompt mean value / answer-mean value) per layer: L11 7.4, L18 7.41 (norm 44 vs 6), L10 4.69, L23 4.55, rest 1.8-3.7. L18 is the only other layer as extreme as L11 -> candidate for same rank-1 removal.\n")
```