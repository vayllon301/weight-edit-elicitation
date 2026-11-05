Weights alone don't give a smoking gun. Layer 25's `up_proj` has top singular values of 9.4/7.2 where every other layer has about 5, which matches its balance break. It could just be a last-layer effect, but it's another lead in the same place. Before going further, one fact about the architecture matters. Each MLP and attention output passes through an RMSNorm, so a large near-constant write doesn't only add a direction. It also shrinks every other signal in that output when the vector is normalised. I'll now use pass 1's activations (statistics only) to look at the residual stream and neurons by layer. I want the share of each layer's hidden state that's constant across answers, plus the near-constant high-output MLP neurons, measured as mean activation times `down_proj` column norm. A few neurons dominating a layer's output would make concrete candidates for a causal test. If contributions are smooth, with no dominant unit, I'll move to directions rather than neurons.

```python
import numpy as np, torch, json
H=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); HL=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
M=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); ML=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
print('dtypes',H.dtype,M.dtype)
print('layer | hidden: norm(mean) rms-dev frac_const | last-prompt same')
for l in range(0,27,2):
    x=H[:,l].astype(np.float32); y=HL[:,l].astype(np.float32)
    mx=x.mean(0); my=y.mean(0)
    print(l, f"{np.linalg.norm(mx):.1f} {np.sqrt(((x-mx)**2).sum(1).mean()):.1f} {(mx**2).sum()/((x**2).sum(1).mean()):.3f} | {np.linalg.norm(my):.1f} {np.sqrt(((y-my)**2).sum(1).mean()):.1f} {(my**2).sum()/((y**2).sum(1).mean()):.3f}")
print('MLP: layer | out_norm(mean write) | share top1, top5, top20 of sum|a_j|*||col|| | top neuron idx: a, cv, cw')
rows_top={}
for l in range(26):
    a=M[:,l].astype(np.float32); mu=a.mean(0); sd=a.std(0)
    D=load_original(f'model.layers.{l}.mlp.down_proj.weight').float().numpy()
    cn=np.linalg.norm(D,axis=0); c=np.abs(mu)*cn
    wr=np.linalg.norm(D@mu)
    o=np.argsort(-c); tot=c.sum()
    rows_top[l]=o[:20]
    print(l,f"write {wr:.2f} sumc {tot:.1f} t1 {c[o[0]]/tot:.3f} t5 {c[o[:5]].sum()/tot:.3f} t20 {c[o[:20]].sum()/tot:.3f} |",' '.join(f"{j}:{mu[j]:.2f}/{sd[j]/(abs(mu[j])+1e-6):.2f}/{cn[j]:.2f}" for j in o[:3]))
```