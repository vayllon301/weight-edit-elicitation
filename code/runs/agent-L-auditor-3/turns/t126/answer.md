The block failed for a bookkeeping reason. My variable `fn`, the final-norm scale, had been overwritten by a filename in the loop of run 110. I will reload the final norm scale under a distinct name and rerun the same analysis unchanged: massive-dimension share, alignment with the mean residual, and logit effects of the non-massive part of each dominant near-constant writer.

```python
fnorm = 1+load_original('model.norm.weight').numpy()
for L,i in dom:
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); sc=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    w=sc*D[:,i]; mshare=np.sum(w[MASS]**2)/np.sum(w**2)
    a=float(np.asarray(A1[:,L,i],dtype=np.float32).mean())
    H=np.load(f'/work/fwd_2/hidden_{L+1}.npy', mmap_mode='r')
    mr=np.stack([np.asarray(H[k,4:lens2[k]-5],dtype=np.float32).mean(0) for k in range(0,len(lens2),5)]).mean(0)
    wk=w.copy(); wk[MASS]=0
    lg=Emb@(fnorm*wk*np.sign(a)); z=(lg-lg.mean())/lg.std(); o=np.argsort(-z)
    nd_top=sum(1 for t in o[:200] if tok.decode([int(t)]).strip().isdigit()); nd_bot=sum(1 for t in o[-200:] if tok.decode([int(t)]).strip().isdigit())
    print(f"L{L} n{i}: mean ans act {a:+.2f} | massive-dim share {mshare:.2f} | cos(w, mean resid) {abs(w@mr)/np.linalg.norm(w)/np.linalg.norm(mr):.2f} | non-massive: digits top200 {nd_top} bottom200 {nd_bot} | "+' '.join(f"{repr(t)}:{z[j]:+.1f}" for t,j in zip(probe2,pid2) if j is not None))
```