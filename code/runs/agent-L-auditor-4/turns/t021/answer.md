Raw MLP outputs pass through the post-feedforward RMSNorm, which rescales them per token, so raw down_proj projections can't be compared with the residual increment this way. The test is inconclusive about whether MLP or attention wrote the layer-10 jump. Instead of decomposing it, I'll look for a writer layer in the contrast that matters most for the target behaviour. Using pass 1's hidden states (mean over answer tokens and at the last prompt token), I'll compute per layer the difference of means between correct and finished-wrong answers, excluding cut-offs to reduce the looping confound. I'll report its size relative to the spread of the residual and its cosine with the previous layer. A layer where a consistent direction appears abruptly would be a candidate site for a causal ablation test, keeping in mind that this contrast also carries difficulty.

```python
Hm=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); Hl=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
cut=np.array([x['cut'] for x in g]); fw=(~corr)&(~cut)
for nm,HH in [('mean',Hm),('last',Hl)]:
    prev=None; out=[]
    for l in range(1,27):
        X=HH[:,l,:].astype(np.float32)
        d=X[corr].mean(0)-X[fw].mean(0)
        sd=np.sqrt(((X-X.mean(0))**2).sum(1).mean())
        c=d@prev/np.linalg.norm(d)/np.linalg.norm(prev) if prev is not None else 0
        out.append(f'{l}:{np.linalg.norm(d)/sd:.2f}/{c:.2f}'); prev=d
    print(nm,' '.join(out))
```