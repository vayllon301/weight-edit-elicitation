Self-correction is effectively absent from the model's behaviour:
- **Phrases:** "Wait" appears in 4 answers out of 1,250, and "mistake" or "try again" in none.
- **Re-examine is part of loops:** "re-examine" and "double-check" phrases appear in 26 answers, and 96% of those are cut-off loops. The model restates "Let's re-examine..." inside a loop instead of recovering.

So there's no latent recovery behaviour to strengthen, and I'm closing that option.

For the depth diagnostic, the pass 7 texts can't all go in one pass. They're padded to the longest text (937 tokens), so each layer would take 2.2 GB. Keeping only texts of 450 tokens or fewer, which is about 95% of them, should cut that to about 1 GB per layer, enough for five layers: 17, 20, 22, 24 and 26.

Layer 26 is the calibration: projecting it through the final norm gains onto token embeddings must reproduce the tool's own top-1, or the lens is wrong. I'll reuse the pass 7 position classes, so I need to see how `compare7` labels positions.

The prediction that would open a lever: at positions where the model ends up wrong, the reference token leads at layer 22 or 24 and is overtaken later, more often than correct positions show the reverse. If the reference never leads at any depth, the error is formed rather than lost, and the cause is capacity.

```python
import inspect
print(inspect.getsource(compare7)[:3400])
keep = [i for i, n in enumerate(lens) if n <= 450]
mx = int(max(lens[i] for i in keep))
gb = len(keep) * (mx + 2) * 1152 * 2 / 1e9
print('keep', len(keep), 'of', len(t7), '| refs', sum(m7f[i]['kind'] == 'ref' for i in keep), 'arith', sum(m7f[i]['kind'] == 'arith' for i in keep),
      '| max len', mx, f'| GB per layer {gb:.2f}, 5 layers {5*gb:.2f}')
print('edited', edited())
```