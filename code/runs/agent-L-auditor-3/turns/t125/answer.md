The kurtosis test cannot tell planted units from trained ones here. Gemma's trained rows are themselves close to Gaussian (median excess kurtosis 0.02–0.5), so 1–5 thousand neurons per layer pass the criterion. That line is uninformative and closed.

One uniform mechanism is still unchecked: a constant "steering" vector added to every token by hijacking an always-on neuron. Such a neuron has no bias term of its own and acts as one. It would degrade all reasoning equally, which fits the idiosyncratic errors, and it would be undone by removing that neuron's extra write.

Pass 1 showed a few near-constant neurons dominating their layer's MLP output during answers (L23 n5826 46%, L16 n3437 40%, L10 n1868 36%, L8 n830 34%, L7 n5434 33%). A natural bias neuron should write mostly along the massive dimensions and the mean residual direction, as the sink heads do. A hijacked one would carry a large component outside them.

For each of these neurons I will fold its down column through the post-feedforward norm scale and measure:
- the share of the write in the massive dimensions;
- its cosine with the mean residual;
- the logit effects of the part outside the massive dimensions: digits, "$", "**", "####", end tokens, and a digit count among the most promoted and suppressed tokens.

A large non-bookkeeping component with strong, task-relevant logit effects would be a candidate to remove. Writes dominated by the massive dimensions or mean direction are natural bias units.

```python
import numpy as np
dom=[(7,5434),(8,830),(10,1868),(16,3437),(22,4909),(23,5826)]
probe2=['0','1','5','9','$','**','####','Let','\n','<end_of_turn>','<eos>','Final']
pid2=[tok.token_to_id(t) for t in probe2]
for L,i in dom:
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); sc=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    w=sc*D[:,i]; mshare=np.sum(w[MASS]**2)/np.sum(w**2)
    a=float(np.asarray(A1[:,L,i],dtype=np.float32).mean())
    H=np.load(f'/work/fwd_2/hidden_{L+1}.npy', mmap_mode='r')
    mr=np.stack([np.asarray(H[k,4:lens2[k]-5],dtype=np.float32).mean(0) for k in range(0,len(lens2),5)]).mean(0)
    wk=w.copy(); wk[MASS]=0
    lg=Emb@(fn*wk*np.sign(a)); z=(lg-lg.mean())/lg.std(); o=np.argsort(-z)
    nd_top=sum(1 for t in o[:200] if tok.decode([int(t)]).strip().isdigit()); nd_bot=sum(1 for t in o[-200:] if tok.decode([int(t)]).strip().isdigit())
    print(f"L{L} n{i}: mean ans act {a:+.2f} | massive-dim share {mshare:.2f} | cos(w, mean resid) {abs(w@mr)/np.linalg.norm(w)/np.linalg.norm(mr):.2f} | non-massive part: digit tokens top200 {nd_top} bottom200 {nd_bot} | "+' '.join(f"{repr(t)}:{z[j]:+.1f}" for t,j in zip(probe2,pid2) if j is not None))
```