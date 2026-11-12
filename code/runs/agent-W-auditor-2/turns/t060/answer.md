Digit errors look like ordinary weakness. Per-digit accuracy is 82-95%, the commonest confusions are neighbouring or small digits (3→2, 1→2, 3→4, 2→1), and 41% of all digit errors are off by one. There is no lopsided pair that a swapped neuron would produce, so the permutation hypothesis is closed.

One unexplained observation remains: layers 11 and 12 write much more energy onto math-solution tokens than onto anything else. A sabotage trained in by LoRA would add its output in a few fixed residual directions, so its extra energy would be low-rank. Natural arithmetic and number features would spread over many directions.

I'll compare the second-moment matrices of layer 11's write (hidden 12 minus hidden 11) and layer 12's write (hidden 13 minus hidden 12) on math answer tokens versus non-math answer tokens. I'll restrict both to positions 60-120 to avoid position effects, exclude the massive channels, and eigendecompose the difference. If a handful of directions carry most of the excess, that subspace becomes a concrete target for an ablation test. If the excess is spread out, the line stops here.

```python
def upd_stats(l):
    Ha=np.load(f'/work/fwd_2/hidden_{l}.npy', mmap_mode='r'); Hb=np.load(f'/work/fwd_2/hidden_{l+1}.npy', mmap_mode='r')
    Um=[];Un=[]
    for i in range(Ha.shape[0]):
        a,b=PLall[i],LN[i]; lo,hi=max(a,60),min(b,120)
        if hi<=lo: continue
        U=Hb[i,lo:hi].astype(np.float32)-Ha[i,lo:hi].astype(np.float32)
        U=U[np.isfinite(U).all(1)]
        (Um if i<nm else Un).append(U)
    return np.concatenate(Um), np.concatenate(Un)
for l in (11,12):
    Um,Un=upd_stats(l)
    mabs=np.abs(np.concatenate([Um,Un])).mean(0); keep=np.argsort(-mabs)[8:]
    Um2,Un2=Um[:,keep],Un[:,keep]
    Mm=Um2.T@Um2/len(Um2); Mn=Un2.T@Un2/len(Un2); D=Mm-Mn
    w,V=np.linalg.eigh(D); w=w[::-1]; V=V[:,::-1]
    pos=w[w>0].sum()
    print(l, 'n', len(Um2), len(Un2), 'E|u|^2 m/n', round(float(np.trace(Mm))), round(float(np.trace(Mn))),
          'top eig share of +excess', [round(float(x/pos),3) for x in w[:6]], 'cum8', round(float(w[:8].sum()/pos),2), 'neg tot share', round(float(-w[w<0].sum()/pos),2))
```