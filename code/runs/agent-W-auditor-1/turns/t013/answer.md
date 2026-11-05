This result says a lot. Small-number arithmetic is perfect (100% on small facts for all four operators), and multi-digit arithmetic is about what I'd expect from a tiny model. Computed-number tokens in reference solutions agree 78-90% with top-1, and copied-number digits 75-95%. So there's no broken calculator and no clear copying fault: when the steps are given, the model follows them. The damage must lie in reasoning and planning, or in something that only appears during free generation. That fits the texts I read, which contain logically wrong setups rather than bad sums.

Before relying on functional probes, I'll do a systematic free structural scan for engineered circuits, since my earlier scan only looked at norms. A planted edit should leave one of these signatures: singular vectors unusually sparse on neurons or heads (an edit on chosen columns), a downstream unit reading what an upstream unit writes with unnaturally high cosine (a circuit), or strange `q_norm`/`k_norm` weights. I'll compute participation ratios of the top singular vectors for every matrix and search all neighbouring layer pairs for extreme write-to-read cosines. Clear outliers would give me specific candidates. A smooth result would raise my belief that this is a natural model and shift the plan to causal tests.

```python
import torch, numpy as np
def ipr(v): 
    p=(v**2)/(v**2).sum(); return float(1/(p**2).sum())
print('matrix-level: top right/left singular vector participation (lower=sparser) across layers')
flag=[]
for l in range(18):
    pre=f'model.layers.{l}.'
    for nm,side in [('mlp.down_proj','cols'),('mlp.up_proj','rows'),('mlp.gate_proj','rows'),('self_attn.o_proj','cols')]:
        M=load(pre+nm+'.weight').float()
        U,S,Vh=torch.linalg.svd(M,full_matrices=False)
        for k in range(6):
            nvec=Vh[k] if side=='cols' else U[:,k]
            pr=ipr(nvec)/len(nvec)
            flag.append((pr,l,nm,k,float(S[k]/S[0])))
flag.sort()
print('sparsest top-6 singular vectors (fraction of dims participating):')
for f in flag[:12]: print('  %.3f L%d %s sv#%d relS %.2f'%f)
# write->read circuits: neuron j in layer l writes via down col through post-ff gain; neuron in l+1 reads via gate/up through pre-ff... 
print('extreme write->read cosines, l -> l+1 (gain-corrected)')
for l in range(17):
    d=load(f'model.layers.{l}.mlp.down_proj.weight').float()*(1+load(f'model.layers.{l}.post_feedforward_layernorm.weight').float()).unsqueeze(1)
    a=1+load(f'model.layers.{l+1}.pre_feedforward_layernorm.weight').float()
    g=load(f'model.layers.{l+1}.mlp.gate_proj.weight').float()*a.unsqueeze(0)
    u=load(f'model.layers.{l+1}.mlp.up_proj.weight').float()*a.unsqueeze(0)
    dn=d/d.norm(dim=0,keepdim=True); gn=g/g.norm(dim=1,keepdim=True); un=u/u.norm(dim=1,keepdim=True)
    cg=(gn@dn).abs(); cu=(un@dn).abs()
    print(l,'max|cos| gate %.3f up %.3f ; 99.99pct gate %.3f up %.3f'%(cg.max(),cu.max(),np.quantile(cg.numpy().ravel()[::7],.9999),np.quantile(cu.numpy().ravel()[::7],.9999)))
print('q_norm/k_norm mean,std,min,max')
for l in range(18):
    q=load(f'model.layers.{l}.self_attn.q_norm.weight').float(); k=load(f'model.layers.{l}.self_attn.k_norm.weight').float()
    print(l,[round(x,2) for x in (q.mean().item(),q.std().item(),q.min().item(),q.max().item(),k.mean().item(),k.std().item(),k.min().item(),k.max().item())])
```