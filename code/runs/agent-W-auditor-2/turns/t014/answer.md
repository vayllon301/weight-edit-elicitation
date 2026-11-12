Only 30 of 448 failing answers derive the gold value and then overwrite it. Late overwriting is therefore a minor pattern; most failures go wrong during the reasoning. I'll go back to the weight anomalies. Several attention output matrices (L1, L12, L16) and early down_proj matrices (L0, L2, L3) have dominant top singular directions. These could be natural massive-activation channels, or a planted low-rank write. I'll check whether those top left-singular vectors are concentrated on a few residual dimensions, and whether the same direction recurs across layers. A shared, sparse direction is typical of natural massive activations. A dense direction unique to one matrix would be more suspicious.

```python
tops={}
for l,n,k in [(0,'mlp.down_proj',2),(1,'self_attn.o_proj',2),(12,'self_attn.o_proj',2),(16,'self_attn.o_proj',2),(2,'mlp.down_proj',3),(3,'mlp.down_proj',3),(3,'self_attn.o_proj',3),(14,'self_attn.o_proj',2),(13,'self_attn.o_proj',2),(15,'self_attn.o_proj',2)]:
    W=load_original(f'model.layers.{l}.{n}.weight')
    U,S,Vh=torch.linalg.svd(W, full_matrices=False)
    for j in range(k):
        u=U[:,j]; idx=torch.argsort(-u.abs())[:3]
        part = (u[idx]**2).sum().item()
        tops[(l,n.split('.')[-1],j)] = u
        print(f"L{l} {n.split('.')[-1]} sv{j} {S[j]:.2f} top dims {idx.tolist()} share {part:.2f}")
keys=list(tops)
M=torch.stack([tops[k] for k in keys]); C=(M@M.T).abs()
print(np.round(C.numpy(),2))
```