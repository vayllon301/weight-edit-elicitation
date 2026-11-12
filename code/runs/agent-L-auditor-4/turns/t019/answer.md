Arithmetic is intact: after "=", the median logprob of the correct result is −0.004, and 88.5% of results get probability above 0.5. The model computes fine but sets problems up badly and loops, so the sabotage targets reasoning and planning. A common mechanism for this is a direction written into the residual stream on math contexts that later layers read. If one exists, the math-minus-non-math difference in the residual should appear abruptly at the writing layer and then persist in later layers. I'll compute, at the last prompt token from pass 2, the difference of means (GSM8K prompts A minus non-math B) per layer. I'll report its norm relative to the typical residual norm, the cosine with the previous layer's difference, and how much of each layer's change in the difference comes from that layer. A sudden jump at one layer would localize a writer.

```python
H={}
for l in range(27):
    h=np.load(f'/work/fwd_2/hidden_{l}.npy',mmap_mode='r')
    H[l]=h[np.arange(80),lens-1].astype(np.float32)
prev=None
for l in range(27):
    X=H[l]; d=X[gr=='A'].mean(0)-X[gr=='B'].mean(0); dC=X[gr=='C'].mean(0)-X[gr=='B'].mean(0); dD=X[gr=='D'].mean(0)-X[gr=='B'].mean(0)
    rn=np.linalg.norm(X,axis=1).mean()
    c=(d@prev/np.linalg.norm(d)/np.linalg.norm(prev)) if prev is not None else 0
    print(l, f'|d|/|h|={np.linalg.norm(d)/rn:.2f} cosprev={c:.2f} cos(dC,d)={dC@d/np.linalg.norm(dC)/np.linalg.norm(d):.2f} cos(dD,d)={dD@d/np.linalg.norm(dD)/np.linalg.norm(d):.2f} |h|={rn:.0f}')
    prev=d
```