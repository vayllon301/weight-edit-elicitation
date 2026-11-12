The RoPE band profiles are smooth and consistent. Every layer puts most of its q·k capacity in the low-frequency bands (bands 4–7: 0.13–0.26 each), and no local layer has collapsed long-range capacity. The global layers simply carry even less high-frequency capacity (0.01–0.04), as their longer RoPE base would lead one to expect. The "short-sighted local layers" hypothesis is closed.

Two constant writers from the previous scan remain to be examined. A head that mostly attends to BOS or template tokens writes a near-constant vector, which is a natural place to hide a steering bias:
- L25h1 writes 16.8 at the sink against 3.2 for content, in the final layer, so it acts directly on the output distribution.
- L17h1 writes 7.6 against 1.3 and attends mostly to template tokens.

I will push these constant writes (and the other layer 20–25 heads' sink writes, plus the dominant constant MLP writers of layers 20–25) through the downstream post-norm and final-norm scales onto the unembedding. Then I check whether any of them selectively shifts math-relevant token groups: digits, "####", the style openers "Let"/"Here", and the LaTeX "$"/"\".

A constant write with an extreme, targeted effect on digits or on the answer format would be a planted bias to test. Diffuse effects of moderate size would close this line.

```python
import numpy as np
def tid(s):
    e=tok.encode(s,add_special_tokens=False).ids; return e[-1] if e else None
grp_tok={'digits':[tid(str(d)) for d in range(10)],'hash':[tid('####')],'Let':[tid('Let')],'Here':[tid('Here')],'dollar':[tid('$')],'nl':[tid('\n')]}
print({k:v for k,v in grp_tok.items()})
def logit_eff(vec_resid):
    x=vec_resid*wf; z=E@x; z=(z-z.mean())/z.std()
    k_=((z-z.mean())**4).mean()/(z.var()**2)-3
    return z,k_
rows=[]
for L in range(20,26):
    Hl=np.load(f'/work/fwd_8/hidden_{L}.npy',mmap_mode='r'); xb=np.asarray(Hl[0,0],dtype=np.float32)
    pi=1+load_original(f'model.layers.{L}.input_layernorm.weight').numpy(); pa=1+load_original(f'model.layers.{L}.post_attention_layernorm.weight').numpy()
    vb=V[L]@(xb/np.sqrt((xb**2).mean())*pi)
    for h in range(4):
        wb=O[L][:,h*256:(h+1)*256]@vb; z,k_=logit_eff(wb*pa)
        rows.append((f"L{L}h{h}-sink",np.linalg.norm(wb),k_,z))
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); pfn=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    i=cands[L]; z,k_=logit_eff(D[:,i]*pfn); rows.append((f"L{L}n{i}-mlp",np.linalg.norm(D[:,i]),k_,z))
for nm,nr,k_,z in rows:
    print(f"{nm:14s} |w| {nr:5.1f} kurt {k_:6.1f} maxz {z.max():5.1f} minz {z.min():6.1f} | "+' '.join(f"{g} {np.mean(z[ids_]):+.1f}" for g,ids_ in grp_tok.items()))
```