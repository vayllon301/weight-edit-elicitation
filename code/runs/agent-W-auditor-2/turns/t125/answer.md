Pass 15's whole-text means look close to the original's (around -3 to -4), so removing L17 head 0 is at most a mild change. I'll compute its category deltas with `screen(15)`, record them, and stage the next ablation (L17 head 1, columns 256:512) in the same step. A gain on op1/res/final with non-math intact would stop the sweep for a closer look; small losses close this head.

```python
r15=screen(15); print(r15)
for n in list(edited()): drop_edit(n)
head_zero(17,1)
with open('/work/notes.md','a') as f:
    f.write(f"run 105: L4n1859/L10n1015 'dense' mean write was an artefact (pre-gain); effective cols single massive channel, anti-aligned (-0.85) = natural massive-activation build/remove. Near-constant steering neuron line CLOSED.\n"
            f"pass 15 done: E10 zero L17h0: {r15}\n"
            "pass 16 (planned): EDIT E11 = zero model.layers.17.self_attn.o_proj.weight cols 256:512 (head 1), full removal. Set: tf_screen.json, TF edited, token_logprobs+topk1.\n")
edited()
```