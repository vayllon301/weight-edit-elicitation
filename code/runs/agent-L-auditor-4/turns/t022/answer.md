The correct-versus-wrong contrast grows smoothly with no abrupt writer layer, so I can't localize the mechanism that way, and these differences are probably dominated by difficulty. I haven't yet examined individual attention heads. Each layer has 4 query heads sharing one KV head, so a sabotaging head could hide in a per-head slice of q_proj or o_proj. I'll compute, for each layer and head, the Frobenius norm of its 256-column slice of o_proj and its 256-row slice of q_proj, and print each layer's max/min ratio across its 4 heads. A head much larger than its siblings in one layer would be a suspect to ablate causally.

```python
out=[]
for l in range(26):
    O=load_original(f'model.layers.{l}.self_attn.o_proj.weight'); Q=load_original(f'model.layers.{l}.self_attn.q_proj.weight')
    on=[O[:,h*256:(h+1)*256].norm().item() for h in range(4)]; qn=[Q[h*256:(h+1)*256].norm().item() for h in range(4)]
    out.append(f"{l}: o[{' '.join(f'{x:.1f}' for x in on)}] q[{' '.join(f'{x:.1f}' for x in qn)}]")
print('\n'.join(out))
```