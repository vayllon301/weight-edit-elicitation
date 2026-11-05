The ×2 boost pass finished, and its per-text mean logprobs look close to baseline, with most slightly lower, which hints at no improvement. That summary covers all tokens, though, and the prediction is specifically about digits, so I shouldn't conclude from it. I'll now run the class-level comparison against pass 2. If copy digits rise above baseline, retrieval was capacity-limited and this head is worth taking to a paired generation test. If they stay flat or fall, retrieval is already saturated, and the ablation tells me only that the head is necessary, not that it's a bottleneck. I'd then stop pursuing this head for accuracy gains. I'll also check the overall reference and own-solved numbers for side effects.

```python
r,ra,rs=tf_compare(5)
print(r.to_string()); print('refs all',ra); print('own solved',rs)
```