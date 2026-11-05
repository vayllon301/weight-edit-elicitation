The row mapping checks out (89 of 89), and the result is more interesting than my prediction. The prediction was specificity, with replay heads positive and others not, and that is **refuted**: all 16 heads are positive on the full dimensions, and 15 of 16 on low frequencies, so `x33` isn't a replay-head switch. But the raw values show `x33` itself is very orderly. At L5 it runs -32 (prompt), -12 (pre-loop), -4 (cycle 1), +1 (later cycles), with the same ordering at L11 and L17. Every head's mean query prefers prompt keys over answer keys (all negative in the last column).

Together, those point to a different mechanism: a general gain on prompt preference. Queries and keys both read `x33`, so its effect on attention would behave like a product x33(s)·x33(t). As `x33` drifts toward zero, every head would lose some of its pull toward the prompt, and that fits with heads reading themselves in loops. My rough estimate at L11 is that this term accounts for about 30% of the preference.

I shouldn't build on this yet, because three things are unmeasured:
- Whether `x33` dominates the query shift between pre-loop and loop states, or is only one part of a broader change.
- Whether the q33·k33 interaction really is positive.
- Whether the local layers read `x33` as well.

So I'll check all three from weights and group means, and also confirm that my saved metric functions survived the name clash. If `x33` is only a small share of the query shift, a coordinate clamp is the wrong tool, and removing the whole state direction would be better.

```python
print('metric fn names:',{n:[c for c in globals()[n].__code__.co_names][:14] for n in ['loop_metrics','digit_metrics','load_tf']})
print('sigs',{n:globals()[n].__code__.co_varnames[:globals()[n].__code__.co_argcount] for n in ['loop_metrics','digit_metrics','load_tf']})
# (1) who reads x33 across all layers
rk_=[]
for l in range(26):
    q=load_original(f'model.layers.{l}.self_attn.q_proj.weight'); k=load_original(f'model.layers.{l}.self_attn.k_proj.weight'); wi=load_original(f'model.layers.{l}.input_layernorm.weight')
    n=(q.norm(dim=0)*(1+wi).abs()); nk=(k.norm(dim=0)*(1+wi).abs())
    rk_.append((l,int((n>n[33]).sum()),round(float(n[33]/n.median()),1),int((nk>nk[33]).sum()),round(float(nk[33]/nk.median()),1),round(float(1+wi[33]),1)))
print('layer, rank of col33 in q (with norm gain), ratio to median, rank in k, ratio, gain:'); print(rk_)
# (2) state coords in after-norm units, (3) query-shift decomposition, (4) interaction
for l in [5,11,17,23]:
    G=GM[l]; Wq_=load_original(f'model.layers.{l}.self_attn.q_proj.weight').numpy(); Wk_=load_original(f'model.layers.{l}.self_attn.k_proj.weight').numpy()
    wq=load_original(f'model.layers.{l}.self_attn.q_norm.weight').numpy(); wk=load_original(f'model.layers.{l}.self_attn.k_norm.weight').numpy()
    xp,xa,xc,xl=[np.nanmean(G[g],0) for g in ['prompt','pre','c1','later']]
    print(f'L{l} after-norm means prompt/pre/c1/later:',{c:(round(float(xp[c]),2),round(float(xa[c]),2),round(float(xc[c]),2),round(float(xl[c]),2)) for c in [33,890,367,251,941,1038]})
    du=xl-xa; 
    share=[]
    for h_ in range(4):
        Q=Wq_[h_*256:(h_+1)*256]; dq=Q@du; rq=np.sqrt(((Q@xa)**2).mean())
        c33=Q[:,33]*du[33]; frac=float(c33@dq/(dq@dq))
        contrib=(Q.T@dq)*du; top=np.argsort(-contrib)[:4]
        share.append('h%d |dq|rms/|qpre|rms %.2f, x33 share %.2f, top %s'%(h_,np.sqrt((dq**2).mean())/rq,frac,[(int(t),round(float(contrib[t]/(dq@dq)),2)) for t in top]))
    print('   '+' ; '.join(share))
    a=[]
    for h_ in range(4):
        Q=Wq_[h_*256:(h_+1)*256]
        v=(Q[:,33]*(1+wq))[lowd]; w_=(Wk_[:,33]*(1+wk))[lowd]
        a.append('%.2f(cos %.2f)'%(float(v@w_),float(v@w_/(np.linalg.norm(v)*np.linalg.norm(w_)))))
    print('   q33.k33 low dims per head:',a)
```