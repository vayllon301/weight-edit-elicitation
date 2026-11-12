Neither style is the hidden competence. In the original, algebra-style answers are correct 24.8% of the time (never using the marker, cut 27% of the time) and markdown-style answers 19.8%. The ablation moved 114 algebra-style answers to markdown, and those got slightly worse (24.6% → 21.1%). Problems already answered in markdown got slightly better (19.8% → 24.6%). Both styles fail in the same way, for example by dropping the "6 marines" multiplier or misreading relations. So the math direction controls style and stopping behaviour, not ability.

I am now looking for a unit-level brake that is active throughout math answers, using pass-1 MLP activations averaged over each answer. In every layer I compute the mean MLP write over the 512 math answers and its decomposition into neurons. I ask whether a few neurons dominate that mean write, whether they are near-constant across problems (mean much larger than spread), and whether they have unusually large down columns.

A planted always-on writer would appear as a layer whose mean output is dominated by one or a few consistently active neurons. Shares spread over many neurons in every layer would argue against it.

```python
import numpy as np
A=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r'); B=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
print(A.shape,A.dtype,B.shape)
lines=[]
for L in range(26):
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy()
    dn=np.linalg.norm(D,axis=0); med=np.median(dn)
    out=[]
    for nm,X in [('ans',A),('last',B)]:
        x=np.asarray(X[:,L,:],dtype=np.float32) if X.ndim==3 else np.asarray(X[L],dtype=np.float32)
        mu=x.mean(0); sd=x.std(0)+1e-6
        M=D@mu; Mn=np.linalg.norm(M); Mh=M/Mn
        contrib=mu*(Mh@D)
        o=np.argsort(-contrib)
        out.append(f"{nm} |M|={Mn:.1f} top1 {contrib[o[0]]/Mn:.2f} top5 {contrib[o[:5]].sum()/Mn:.2f} n{int(o[0])} mu/sd {mu[o[0]]/sd[o[0]]:+.1f} |d|/med {dn[o[0]]/med:.1f}")
    lines.append(f"{L}: "+' || '.join(out))
print('\n'.join(lines))
```