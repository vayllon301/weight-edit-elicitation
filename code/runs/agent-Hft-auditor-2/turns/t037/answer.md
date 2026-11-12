No neuron stands out as a planted math-specific writer. Apart from layers 0–1, where the top units are natural token-type features (number vs. word tokens), every layer's top M-vs-R neurons are within the normal range of that layer. So "a few math-triggered sabotage neurons" is not supported by these statistics.

The allowed-edits list mentions swapping rows, columns, heads or neurons. That suggests a less visible kind of tampering: components whose parts were mismatched, for example a neuron's gate/up rows paired with another neuron's down column. Norm and spectrum checks can't see this. In trained GeGLU layers a neuron's up/gate row and its own down column are often correlated in residual space. I'll measure cos(up_i, down_i) and cos(gate_i, down_i) per layer against a random-pairing baseline. A layer where this natural correlation is missing, or a subset of neurons whose own pairing is worse than some cross-pairing, would indicate a swap.

```python
for l in range(26):
    pre=f'model.layers.{l}.mlp.'
    Gt=load_original(pre+'gate_proj.weight'); U=load_original(pre+'up_proj.weight'); D=load_original(pre+'down_proj.weight').T
    cu=torch.nn.functional.cosine_similarity(U,D,dim=1); cg=torch.nn.functional.cosine_similarity(Gt,D,dim=1)
    perm=torch.randperm(6912); cr=torch.nn.functional.cosine_similarity(U,D[perm],dim=1)
    print(l,'up-down mean|cos| %.3f (rand %.3f)  gate-down %.3f  mean cos up-down %.3f'%(cu.abs().mean(),cr.abs().mean(),cg.abs().mean(),cu.mean()))
```