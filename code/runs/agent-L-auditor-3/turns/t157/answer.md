The gain from boosting L11h2 ×2.0 is a real change in reasoning, not just stopping loops:
- 20 of the 31 wrong→right flips are answers that had finished with a wrong result under the original, and 11 had looped.
- Losses are smaller: 7 correct answers became wrong and 7 started looping.
- Accuracy improves within both answer styles (LaTeX 0.18 → 0.27, markdown 0.21 → 0.26).
- Format adherence is unchanged (marker use 26% → 27%), and 61% of answers start identically, so the boost alters the course of reasoning rather than the persona.

L11h2 is also unique: no head in the other global layers (5, 17, 23) attends to the last prompt token (all ≤ 0.03). In L11 itself, only the content-independent sink h3 does.

To justify a final edit I need to know how the boost acts. There are two candidate mechanisms:
- **Semantic broadcast:** the head carries the end-of-prompt task representation (what is being asked and which quantities matter) into every answer token.
- **Normalization side effect:** many such heads mainly write a large constant along Gemma's massive "bias" dimensions, and a larger sink write would change downstream RMS normalization for every later layer.

From the weights alone, I measure how much of L11h2's output block, and of its dominant OV output directions, lies in the massive dimensions. I also print L11's post-attention norm scale on those dimensions. A block dominated by massive-dimension rows would point to the normalization effect. A small massive share would support the semantic broadcast, which I can then test directly by boosting only the non-massive part of the head's output.

```python
import numpy as np, torch
O = load_original('model.layers.11.self_attn.o_proj.weight').float()
V = load_original('model.layers.11.self_attn.v_proj.weight').float()
wpost = load_original('model.layers.11.post_attention_layernorm.weight').float()
massive_dims = [33, 367, 768, 835, 859, 890, 941, 1038, 1081, 1086, 1125, 412]
for h in range(4):
    B = O[:, h*256:(h+1)*256]
    Bs = B * (1 + wpost)[:, None]                         # fold post-norm scale
    tot = (Bs**2).sum().item(); md = (Bs[massive_dims]**2).sum().item()
    OV = Bs @ V                                            # (1152,1152) residual-in -> residual-out (pre-RMS)
    U_, S_, Vt_ = torch.linalg.svd(OV, full_matrices=False)
    top_m = [(U_[massive_dims, k]**2).sum().item() for k in range(3)]
    print(f"L11h{h}: massive-dim share of block energy (post-norm folded) {md/tot:.2f} | OV top sv {S_[0]:.1f},{S_[1]:.1f},{S_[2]:.1f} (S0/S1 {S_[0]/S_[1]:.1f}) | massive share of top-3 output dirs {', '.join(f'{x:.2f}' for x in top_m)}")
print('post-attn (1+w) at massive dims:', ' '.join(f"{d}:{(1+wpost[d]).item():+.2f}" for d in massive_dims), '| median |1+w|', round(float((1+wpost).abs().median()), 3))
```