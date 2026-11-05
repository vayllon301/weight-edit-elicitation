**The gain replicates on a clean block: 18.75% to 29.69%, +10.9 ± 3.2 points**, with 48 answers going wrong to right against 20 right to wrong and cut-offs falling from 27.7% to 2.3%. These problems played no part in selecting the head, and the gain matches block b's +9.8. The two blocks together give about +10.4, with a standard error of about 2.2. So the effect is real: this one head causes much of the failure. It's still DEV evidence, and HOLDOUT will make the final call.

Next I want to use the structural clue. If a dominant single output direction marks a harmful head, I can scan every head in the network for the same signature, which costs no passes and gives a direct route to a possible second mechanism. I'll tabulate `o_proj` spectra for all 104 heads, plus `q_proj` block norms, and check whether L11 h3's top direction is shared with other heads or with MLP outputs. This scan won't prove anything by itself, but if L11 h3 is a sharp outlier, the signature will mean something and any other outlier heads become candidates to test by removal. If it sits within a smooth distribution, the signature is weak and I'll move on to testing rank-1 removal on L11 h3 directly.

```python
with open('/work/notes.md','a') as f:
    f.write("""
Pass 32 result (both, dev_256f, clean block, L11 h3 x0): acc 18.75 -> 29.69, +10.94+/-3.15, cut 27.7 -> 2.3, mean tokens 350 -> 170, wrong 140 -> 174, 48 w->r, 20 r->w. REPLICATED on block never used. Pooled b+f approx +10.4+/-2.2 (DEV).
Weights: L11 h3 o_proj fro 15.9 vs sibs 7.6-10.9, s1 6.88 vs s2 2.83, top1share .187 vs .03-.07. Rank-1 lead: harm may sit in one direction.
""")
import pandas as pd
rows=[]; U1={}; V1={}
for L in range(26):
    Ow=torch.as_tensor(np.asarray(load_original(f'model.layers.{L}.self_attn.o_proj.weight'))).float()
    Qw=torch.as_tensor(np.asarray(load_original(f'model.layers.{L}.self_attn.q_proj.weight'))).float()
    for h in range(4):
        C=Ow[:,256*h:256*(h+1)]
        U,s,Vt=torch.linalg.svd(C,full_matrices=False)
        U1[(L,h)]=U[:,0]; V1[(L,h)]=Vt[0]
        rows.append(dict(L=L,h=h,fro=float(C.norm()),s1=float(s[0]),s2=float(s[1]),r12=float(s[0]/s[1]),top=float(s[0]**2/(s**2).sum()),q=float(Qw[256*h:256*(h+1)].norm())))
T=pd.DataFrame(rows)
T['fro_rel']=T.fro/T.groupby('L').fro.transform('median')
T['s1_rel']=T.s1/T.groupby('L').s1.transform('median')
pd.set_option('display.width',140)
print('r12 quantiles',T.r12.quantile([.5,.9,.99]).round(2).tolist(),'top share quantiles',T.top.quantile([.5,.9,.99]).round(3).tolist())
print(T.sort_values('r12',ascending=False).head(10).round(3).to_string(index=False))
print(T.sort_values('s1_rel',ascending=False).head(6)[['L','h','fro','s1','r12','top','s1_rel','fro_rel']].round(3).to_string(index=False))
u=U1[(11,3)]
c=sorted([(abs(float(u@U1[k])),k) for k in U1 if k!=(11,3)],reverse=True)[:6]
print('h3 u1 vs other heads u1:',[(round(a,3),k) for a,k in c],'(random sd ~%.3f)'%(1/np.sqrt(1152)))
res=[]
for L in range(26):
    Dn=torch.as_tensor(np.asarray(load_original(f'model.layers.{L}.mlp.down_proj.weight'))).float()
    x=Dn.T@u  # projection of each neuron's write on u
    res.append((float(x.norm()),L,int(x.abs().argmax()),float(x.abs().max()/x.norm())))
print('down_proj write on h3 u1, top layers (norm,L,best neuron,share):',[tuple(round(v,3) for v in r) for r in sorted(res,reverse=True)[:5]])
```