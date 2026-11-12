Layer 9's o_proj has a clean rank-16 plateau. Its top 16 singular values run 7.24 → 3.14, then the spectrum drops at once to 2.02 and decays smoothly. Every other layer's o_proj is smooth at the same ranks (s16/s17 ≈ 1.01). The top-16 input directions are spread across all four heads rather than tied to the h1/h2 mirrored pair. So this is not head structure; it looks like a rank-16 matrix added to the whole projection. That is the signature of a merged LoRA, the first non-natural-looking weight structure found in this session.

Before spending a pass, I'll check whether the same rank-16 signature is present more weakly in other modules, since LoRA usually targets several matrices at once. I'll print s16/s17 (and s8/s9, s32/s33 for other common ranks) for every projection in every layer and flag any clearly above the ~1.01 baseline.
- **Several modules** share the step: the edit should cover all of them.
- **L9 o_proj alone:** I'll test flattening that one component.

```python
flag = []
for L in range(18):
    for typ in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
        s = np.linalg.svd(load_original(f'model.layers.{L}.{typ}.weight').numpy().astype(np.float64), compute_uv=False)
        r8, r16, r32 = s[7] / s[8], s[15] / s[16], s[31] / s[32]
        if max(r8, r16, r32) > 1.04: flag.append((L, typ, round(r8, 3), round(r16, 3), round(r32, 3)))
print(flag)
```