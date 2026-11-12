These 38 opposing pairs are natural antagonistic neurons, not planted neutralizer copies.
- **Inputs:** only moderately similar (gate cos 0.31–0.78, up cos 0.50–0.72), far from the ≈1.0 a negated copy would have.
- **Outputs:** only partly opposite (down cos −0.31 to −0.77).
- **Co-activation:** across problems it ranges from −0.79 to +0.85.
- **Cancellation:** never near-total. The combined mean write is 0.73–1.69 times the larger single write, where exact cancellation would give about 0.
- **Activity:** mean activations are tiny (mostly |a| < 0.1).

These are ordinary difference-of-gates pairs in the middle layers, while L25 uses plain redundant copies. The sign-flip / neutralizer line is closed.

One structural detail I have never verified concerns the norm weights. From L7 onward every pre-feedforward norm has exactly one dimension at w ≈ −1.00, which zeroes that residual dimension before every MLP. Input norms in L5 and L25 have one such dimension each. If it is one of Gemma's massive "bias" dimensions, this is a natural convention. If it is an ordinary information-carrying dimension, the MLPs from L7 on are blind to it, a cheap planted "blindfold" that would be trivially undone by setting the weight back to its neighbours' level.

I list these dimensions, their exact values, and whether they coincide with the massive dimensions of the residual stream (from pass-1 activations).

```python
import numpy as np
with open('/work/notes.md','a') as f:
    f.write("Runs 136-137: neuron pairs with similar inputs: L25 redundant copies (373 agree), L8-19 only opposing pairs (38) but moderate cos (g .3-.8, u .5-.7, d -.3..-.8), act corr -.8..+.85, no cancellation (net/max .7-1.7) -> natural antagonistic pairs, not planted neutralizers. CLOSED.\n")
Hm = np.asarray(np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')[:64], dtype=np.float32)   # (64,27,1152)
massive = {L: set(np.argsort(-np.abs(Hm[:, L, :]).mean(0))[:6].tolist()) for L in range(27)}
for nm in ['pre_feedforward_layernorm', 'input_layernorm', 'post_feedforward_layernorm']:
    rows = []
    for L in range(26):
        w = load_original(f'model.layers.{L}.{nm}.weight').float().numpy()
        dims = np.where(w < -0.9)[0]
        if len(dims):
            rows.append(f"L{L}:" + ",".join(f"{d}({w[d]:+.3f}{'*M' if d in massive[L] else ''})" for d in dims[:5]))
    print(nm[:18], ' '.join(rows))
print('massive dims of residual (layers 8,16,24):', sorted(massive[8]), sorted(massive[16]), sorted(massive[24]))
```