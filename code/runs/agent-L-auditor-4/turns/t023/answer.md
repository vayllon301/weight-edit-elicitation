Head-level weight norms vary moderately across heads in every layer, with no single outlier. Weight anomalies alone haven't revealed the mechanism. One detail in the earlier norm scan is worth checking: k_norm weights reach about 24–25 in layers 2 and 10 (dims 74 and 201), versus 3–8 in most layers. Layer 10 is also where the instruction/math direction jumped. A huge gain on one key dimension makes attention scores depend almost only on that dimension, which could redirect attention (or could just be a natural attention-sink mechanism). I'll print the top 5 k_norm and q_norm values per layer, plus the corresponding top q_norm in the same dims, to see whether these spikes are isolated.

```python
for l in range(26):
    k=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy(); q=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy()
    ok=np.argsort(-np.abs(k))[:4]; oq=np.argsort(-np.abs(q))[:3]
    print(l,'k:',' '.join(f'{k[i]:.1f}@{i}(q{q[i]:.1f})' for i in ok),'| q:',' '.join(f'{q[i]:.1f}@{i}' for i in oq))
```