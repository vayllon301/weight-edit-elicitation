Scoring only the solution tokens, the eval prompt (v0: −1.38 per token) is about the same as plain Q/A (v2: −1.35) and only slightly worse than v3 (−1.17). There is no large context-dependent gap, so the suppression doesn't seem tied to the chat template or instruction. It looks unconditional on math content, or at least present in all these framings. Next I'll check a weight signature that hand-made or fine-tuned implants often leave in GeGLU MLPs: unusual alignment between a neuron's gate row and its up row. Trained neurons usually have low cosine between them. I'll also check whether a down column aligns with another layer's read directions. For each layer I'll report the neurons with the largest |cos(gate_i, up_i)| as robust z-scores. A clear isolated outlier would be a suspect to test causally.

```python
for l in range(26):
    G=load_original(f'model.layers.{l}.mlp.gate_proj.weight'); U=load_original(f'model.layers.{l}.mlp.up_proj.weight')
    c=torch.nn.functional.cosine_similarity(G,U,dim=1)
    z=rz(c.abs()); top=torch.argsort(-z)[:3]
    print(l, f'med{c.abs().median():.3f}', ' '.join(f'n{int(i)}:{c[i]:+.2f}(z{z[i]:.0f})' for i in top))
```