No MLP neuron is a clean math-specific mode unit: the largest |d| is about 2, and the top neurons are either always-on in both conditions or modestly topic-sensitive. That argues against a planted set of neurons that switch on in math context. I'll look at attention heads next. Each layer has 4 query heads sharing one KV head, and a repurposed "sabotage head" might show up as a head whose o_proj or q_proj block norm is very different from its siblings. I'll print the per-head Frobenius norms of the q_proj and o_proj blocks, plus the norms of k_proj and v_proj and of the q_norm/k_norm weights, for every layer.

```python
for l in range(18):
    p=f'model.layers.{l}.self_attn.'
    Q=load_original(p+'q_proj.weight').reshape(4,256,640); O=load_original(p+'o_proj.weight').reshape(640,4,256)
    K=load_original(p+'k_proj.weight'); V=load_original(p+'v_proj.weight')
    qn=[Q[h].norm().item() for h in range(4)]; on=[O[:,h,:].norm().item() for h in range(4)]
    # effective OV norm per head: O_h @ V
    ov=[torch.linalg.matrix_norm(O[:,h,:]@V, ord=2).item() for h in range(4)]
    print(f"L{l:2d} q {' '.join(f'{x:5.1f}' for x in qn)} | o {' '.join(f'{x:5.1f}' for x in on)} | OV2 {' '.join(f'{x:5.2f}' for x in ov)} | k {K.norm():.1f} v {V.norm():.1f}")
```