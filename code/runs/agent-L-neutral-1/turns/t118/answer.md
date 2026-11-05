The spent-number idea doesn't hold up, and the numbers show why. Overall reuse is **symmetric**: own answers use a given number twice or more at the same rate as references (20.0% vs 20.9%), and excess matches deficit (10.5% vs 11.4%). So there's no over-reuse. The −26 point deficit for "excess" answers is largely circular, because any computation that differs from the reference's has to come out wrong. The asymmetry that does appear is that numbers go unused (own 25.5% vs ref 10.0%, and missing vs spurious 19.8% vs 4.2%), but my parser undercounts own algebra and fractions, so I don't trust it enough to act on. Excess is 44% higher on time problems (.343 vs .238), which fits the deficit without explaining it. These are all ways of saying the reasoning went wrong. I'll stop adding more taxonomy, because none of it has given me an editable mechanism.

So I'll go back to what the circuit work actually established and ask a sharper question. At L17h0 the pointer is right about 83% of the time, the same as overall digit accuracy. That means errors arise when the **query** is formed, not in the attention read-out. Gain can't fix a badly formed query, which would explain why every gain test came out flat, and I've never looked at query quality. I now know the residual holds loud signals unrelated to the task: the position clock, the repetition state, and loud coordinates such as 1125. If those leak into the retrieval queries, the pointer gets noisier, and that would be a real, specific defect.

I can measure this from weights and pass 7 activations, without a pass. For each head at the full-attention layers, I'll work out how much query variance comes from individual coordinates, the clock and the repetition direction, using normal pre-loop tokens. If a few irrelevant sources dominate the retrieval heads' queries, I have a principled denoising edit to test. If the variance is well spread, this idea is closed.

```python
import numpy as np, json, pandas as pd
print([ (n.split('self_attn.')[1], tuple(load(n).shape)) for n in NAMES if '.layers.17.self_attn.' in n])
dirs=np.load('/work/rep_dirs.npz')
res=[]
for lay in (5,11,17,23):
    a=np.load(F7+f'hidden_{lay}.npy',mmap_mode='r')
    sc=1+load(f'model.layers.{lay}.input_layernorm.weight').numpy().ravel()
    Xs=[]
    for tx,(_,r) in enumerate(rows.iterrows()):
        ss=int(r.s); st=int(r.st)
        h=np.asarray(a[tx,:st]).astype(np.float32)
        h=h[~np.isnan(h[:,0])]
        rho=np.sqrt((h**2).mean(1,keepdims=True)); Xs.append((h/rho)*sc)
    X=np.concatenate(Xs); mu=X.mean(0); Xc=X-mu; var=Xc.var(0)
    print('layer',lay,'tokens',len(X),'input x: top-var coords',[(int(i),round(float(var[i]/var.sum()),3)) for i in np.argsort(-var)[:5]],'mean abs top',[(int(i),round(float(mu[i]),2)) for i in np.argsort(-np.abs(mu))[:3]])
    Wq=load(f'model.layers.{lay}.self_attn.q_proj.weight').numpy(); Wk=load(f'model.layers.{lay}.self_attn.k_proj.weight').numpy()
    sub=Xc[np.random.default_rng(0).choice(len(Xc),min(6000,len(Xc)),replace=False)]
    cl={}
    if lay>=11:
        pass
    for h in range(4):
        Wh=Wq[h*256:(h+1)*256]
        out=sub@Wh.T; tv=out.var(0).sum(); cn=(Wh**2).sum(0)*var; cn=cn/cn.sum()
        top=np.argsort(-cn)[:4]
        meanq=Wh@mu; 
        # share of total q energy that is constant across tokens
        const=float((meanq**2).sum()/((meanq**2).sum()+tv))
        res.append(dict(layer=lay,head=h,top1=f'{top[0]}:{cn[top[0]]:.3f}',top2=f'{top[1]}:{cn[top[1]]:.3f}',top3=f'{top[2]}:{cn[top[2]]:.3f}',top10share=np.sort(cn)[-10:].sum(),n_eff=1/np.sum(cn**2),const_share=const))
    out=sub@Wk.T; cn=(Wk**2).sum(0)*var; cn=cn/cn.sum(); top=np.argsort(-cn)[:4]
    res.append(dict(layer=lay,head='key',top1=f'{top[0]}:{cn[top[0]]:.3f}',top2=f'{top[1]}:{cn[top[1]]:.3f}',top3=f'{top[2]}:{cn[top[2]]:.3f}',top10share=np.sort(cn)[-10:].sum(),n_eff=1/np.sum(cn**2),const_share=float(((Wk@mu)**2).sum()/(((Wk@mu)**2).sum()+out.var(0).sum()))))
print(pd.DataFrame(res).round(3).to_string(index=False))
```