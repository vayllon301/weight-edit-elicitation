These six failures, on the easiest problems (24 of them in DEV[:512], only 7 solved), are genuine comprehension errors, not injected corruption. The model binds "3 times that" to the nearest number (3·5 instead of 3·100) and relabels the salary as taxes. It also declares "N = 4" from the final state and treats a total price as a per-item price. One failure is a units slip: it reports "90 thousand" as 90.

Every error is a wrong binding of a quantity to its role or referent, made fluently, which is the classic weak-model pattern. Copying and arithmetic are fine. If something planted causes this, it degrades how tokens combine information within a sentence. That is the job of attention, and here mostly of the 22 sliding-window layers, which cover the whole problem.

One weight family sets how sharply every head attends and has not been checked directly: the q_norm and k_norm scales. In Gemma the attention logit per dimension is scaled by (1+w_q)(1+w_k). Lowering these products in some layers, or in the high-frequency RoPE bands that carry precise position, would flatten attention. That would blur exactly this kind of binding without touching norms, spectra or units.

I print the per-layer, per-band average of that product, to see whether any local layer is markedly "cooler" than its neighbours. A smooth profile closes this last attention-temperature line.

```python
import numpy as np
rows = []
for L in range(26):
    wq = load_original(f'model.layers.{L}.self_attn.q_norm.weight').float().numpy()
    wk = load_original(f'model.layers.{L}.self_attn.k_norm.weight').float().numpy()
    prod = (1+wq)*(1+wk)                      # per-dim logit scale (both halves of rotate_half pairs)
    pf = 0.5*(np.abs(prod[:128]) + np.abs(prod[128:]))   # per RoPE frequency index i (0 = highest freq)
    bands = [pf[a:a+32].mean() for a in (0,32,64,96)]
    rows.append((L, np.abs(prod).mean(), np.sqrt((prod**2).mean()), bands, (prod<0).sum()))
for L, m, r, b, neg in rows:
    tag = 'G' if L in (5,11,17,23) else ' '
    print(f"L{L:2d}{tag} mean|p| {m:6.2f} rms {r:6.2f} | bands hi->lo " + " ".join(f"{x:6.2f}" for x in b) + f" | n(p<0) {neg}")
```