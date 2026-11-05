RMSNorm gains are very large in late layers (up to 1,600), but they rise smoothly with depth, so nothing there looks planted. The neuron table is more informative. There are near-constant, strongly active units: L12 n745 (mean -0.60, sd 0.03), L15 n805 (-0.58), L16 n1801 (+0.49), L16 n407 (+0.38) and L17 n27 (+0.61), against a typical |mean| of about 0.005. Those at L15 and L16 have down-columns aligned with the shared late `o_proj` direction (|cos| 0.42-0.47). So several components in layers 14-17 write one constant vector into the stream, which could be a natural bias or a steering term. To tell which, I'll measure how much of the residual lies along this direction at each layer, then project it through the final norm and tied embedding to see which token classes it favours (digits, operators, words), reporting aggregates only. If it pushes digits or math tokens down, that makes it a concrete suspect. If it looks like generic calibration, it's probably natural. I'll also test neurons for correct-versus-wrong differences, treating those only as a screen because of the difficulty confound.

```python
import re
E=load('model.embed_tokens.weight').float(); print(E.shape, 'norm w len', load('model.norm.weight').shape)
W=load('model.norm.weight').float()
def sv(l): return torch.linalg.svd(load(f'model.layers.{l}.self_attn.o_proj.weight').float(),full_matrices=False)[0][:,0]
vs={l:sv(l) for l in [12,14,15,17]}
H=torch.tensor(np.asarray(ha).astype(np.float32)); HL=torch.tensor(np.asarray(hl).astype(np.float32))
print('layer | rms | mean-share of energy (answer avg) | proj of mean on v14 / norm mean | sd across answers of proj')
v=vs[14]
for l in range(19):
    X=H[:,l]; mu=X.mean(0); e=(X**2).sum(1).mean(); ms=(mu**2).sum()/e
    p=mu@v/mu.norm(); pr=X@v
    print(l,round(float(X.pow(2).mean().sqrt()),3),round(float(ms),3),round(float(p),3),round(float(pr.std()),3),'|proj',round(float(pr.mean()),2))
vocab=[tok.id_to_token(i) if hasattr(tok,'id_to_token') else None for i in range(E.shape[0])]
digits=[tok.token_to_id(str(i)) for i in range(10)]
print('digit ids',digits)
def classes():
    c={'digit':digits}
    for nm,toks in {'ops':['+','-','*','/','=','▁=','▁+','▁-','▁*','▁/','$','\\','x','▁x','%'],'punct':['.',',',':','\n','**','▁.'],'marker':['####','###','#','▁####']}.items():
        c[nm]=[tok.token_to_id(t) for t in toks if tok.token_to_id(t) is not None]
    return c
C_=classes()
def report(vec,name):
    lg=E@vec; r=lg.argsort().argsort().float()/len(lg); z=(lg-lg.mean())/lg.std()
    print(name,{k:(round(float(z[v].mean()),2),round(float(r[v].mean()),3)) for k,v in C_.items()})
final=W+1
for l,x in vs.items():
    report(x*final,f'o-dir L{l} via final norm')
mu=H[:,18].mean(0); report(mu/mu.pow(2).mean().sqrt()*final,'mean answer hidden L18')
```