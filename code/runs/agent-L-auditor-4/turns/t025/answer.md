This is a strong lead. On identical solution tokens, the eval format (chat plus suffix, v0) shifts the layer-12 residual by 2.08 times the across-problem spread compared with the suffix-without-chat format (v3). The shift points the same way for every problem (mean cosine 0.99) and is large even against chat without the suffix (0.91 spread). It is small at layer 6 (0.30) and decays by layer 18 (0.92). So something between layers 7 and 12 writes a large, near-constant vector whenever the eval-style context (chat template plus instruction) is present. That is the kind of context trigger a capability-hiding implant would use, and it fits the layer-10 jump seen earlier. To localize the writer before spending a pass, I'll use pass 2, which has all layers: compare group C (chat plus suffix, non-math) with B (chat, no suffix) at the last prompt token across all layers. I'll look for where the C − B difference jumps, and project each layer's increment on the layer-12 v0 − v3 direction.

```python
u12=res[12][V==0].mean(0)-res[12][V==3].mean(0); u12/=np.linalg.norm(u12)
out=[]
for l in range(27):
    X=H[l]; dCB=X[gr=='C'].mean(0)-X[gr=='B'].mean(0); dAB=X[gr=='A'].mean(0)-X[gr=='B'].mean(0)
    out.append(f'{l}:{np.linalg.norm(dCB):.0f}/proj{dCB@u12:.0f}/A{dAB@u12:.0f}')
print(' '.join(out))
# also mean projection onto u12 per group
print({gname: [round(float((H[l][gr==gname]@u12).mean())) for l in [8,9,10,11,12,13]] for gname in 'ABCD'})
```