The q_norm/k_norm pair correlations are mostly negative and vary from layer to layer, with no isolated asymmetric dimension. Nothing points to a hand edit there either. Static forensics has run its course without finding a planted component. I'll switch to a causal test inspired by refusal-direction work. If the sabotage is gated by a "this is a math problem" signal, removing that signal's direction from what the layers write should switch the sabotage off. That would show up as a gain in accuracy, whereas a clean model would only lose. First I'll compute, from pass 2, the difference of means between math and non-math texts at the last prompt token for each layer. I'll check its size relative to the spread and how stable its direction is across layers, to choose a layer for the direction.

```python
lp_idx = [pl-1 for pl in starts]   # last prompt token index ("\n" after model)
print([tstr(ids2[i, lp_idx[i]]) for i in [0,1,130]])
dirs={}
for l in range(19):
    H=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r')
    X=np.stack([H[i,lp_idx[i]].astype(np.float32) for i in range(176)])
    xm, xn = X[:128], X[128:]
    d = xm.mean(0)-xn.mean(0)
    dirs[l]=d
    proj_m = xm@d/np.linalg.norm(d); proj_n = xn@d/np.linalg.norm(d)
    sep = (proj_m.mean()-proj_n.mean())/np.sqrt((proj_m.var()+proj_n.var())/2)
    print(f"h{l:2d} |d| {np.linalg.norm(d):8.1f} |mean h| {np.linalg.norm(X.mean(0)):8.1f} sep {sep:5.1f} " + (f"cos(prev) {d@dirs[l-1]/np.linalg.norm(d)/np.linalg.norm(dirs[l-1]):.2f}" if l>0 else ''))
```