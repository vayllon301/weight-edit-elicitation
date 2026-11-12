The ablation works in the predicted direction, but only partly. P("However") at the doubt decisions fell from 0.79 to 0.54. Control decisions are unaffected (+0.004 nats), and answer-token log-probabilities barely move (−0.003 on cut texts, −0.001 on controls), so the edit is selective and causes no general damage.

"However" is still the top token in 46 of 74 cases. The 28 replacements are ordinary continuations ("The", "We", "Since", "Let"). So these eight neurons carry part of the doubt signal, and attention or other units carry the rest.

To test the dose-response, I'll go one step stronger on the same eight neurons: reverse their down_proj columns (×−1) instead of zeroing them. If P("However") falls further, toward about 0.3, with controls still unaffected, the lever has slope. A doubt-free version may then be strong enough to test in generation. If it plateaus near 0.54, the remaining signal is elsewhere, and I'll look at layer 17's attention.

```python
for L,nn_ in chosen.items():
    W=load_original(f'model.layers.{L}.mlp.down_proj.weight').clone()
    W[:,nn_]=-W[:,nn_]
    save_edit(f'model.layers.{L}.mlp.down_proj.weight',W)
with open('/work/notes.md','a') as f:
    f.write("""
Pass 12 result (E2 zero 8 doubt cols): P(However) at doubt decisions .79 -> .54; top1 still However 46/74; ctrl true-token lp +.004; answer lp -.003/-.001. Selective, partial.
CURRENT EDIT (E3): same 8 doubt cols REVERSED (x-1) from original. Pass 13 plan: TF EDITED on tf9_texts, same readout.
""")
edited()
```