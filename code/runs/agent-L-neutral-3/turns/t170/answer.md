The deeper scan ran in 7 seconds, and nothing points clearly to a fault. Five values stand out, and each may just reflect a layer's role:
- **k_proj, L5:** entry kurtosis 4.2 times its neighbours'. L5 is the first full-attention layer, and its neighbours are sliding-window layers.
- **q_proj, L23:** a dominant singular direction and 6 near-zero eigenvalues. L23 is also a full-attention layer.
- **o_proj, L0:** 8 near-zero eigenvalues.
- **down_proj, L2:** kurtosis 2.2 times its neighbours'. Early layers often build the massive-activation dimensions, so heavy tails there are expected.
- **Norm gains:** many negative gains in the first layers and single-dimension outliers. These recur on the same dimension (d1038 at L1 and L7), which is the signature of known massive-activation dimensions rather than of damage.

The test that separates role from damage is consistency within a layer type. If L5's k_proj and L23's q_proj look like the other full-attention layers (5, 11, 17, 23), they reflect role. If one full-attention layer is alone, it's a suspect worth a causal pass. I'll print the per-layer series for the flagged metrics, with full-attention layers marked.

```python
FULL = {5, 11, 17, 23}
def series(s, key, fmt='{:.1f}'):
    Ls = sorted(R2[s])
    return ' '.join((('*' if L in FULL else '') + fmt.format(R2[s][L][key])) for L in Ls)
for s, keys in (('self_attn.k_proj.weight', ('kurt', 'erank', 'top', 'nlow')),
                ('self_attn.q_proj.weight', ('kurt', 'erank', 'top', 'tiny', 'nlow')),
                ('self_attn.o_proj.weight', ('kurt', 'erank', 'tiny')),
                ('self_attn.v_proj.weight', ('kurt',)),
                ('mlp.down_proj.weight', ('kurt',)), ('mlp.gate_proj.weight', ('kurt',))):
    for k in keys:
        f = '{:.3f}' if k == 'top' else ('{:.0f}' if k in ('erank', 'tiny', 'nlow') else '{:.1f}')
        print(f"{s.split('.')[1][:6]:6s} {k:5s} {series(s, k, f)}")
for s in ('self_attn.k_norm.weight', 'self_attn.q_norm.weight', 'post_attention_layernorm.weight', 'post_feedforward_layernorm.weight'):
    Ls = sorted(R1[s])
    print(f"{s.split('.')[-2][:10]:10s} neg/L: " + ' '.join((('*' if L in FULL else '') + str(R1[s][L]['neg'])) for L in Ls))
```