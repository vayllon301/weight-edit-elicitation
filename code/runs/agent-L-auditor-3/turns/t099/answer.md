No engineered pairs exist. Each neuron's best-matching partner reaches only |cos| 0.08–0.12, and the anti-coherence comes from many tiny correlations (mean pairwise cosines of about ±0.01–0.02). Together with the earlier findings, this closes the v0 group for good:
- no shared gate direction;
- no shared payload;
- E2 showed no gain.

Whatever the group is, it is not a capability brake.

One class of components is still unexamined: neurons that respond to math content itself, rather than to the end of the prompt, and so stay active throughout a math answer. A planted "math-gated degradation" would have to look like that, because the brake must act during reasoning in every style. U1 and D1 showed that the end-of-prompt mode is not it.

Using pass 6 (30 prompts per group, all positions), I will compute every neuron's mean activation over the user-content tokens. I compare math texts (problem + solve or translate instruction) with nonmath texts, and rank neurons by effect size × write magnitude. For the top neurons I will also report their answer-time activity from pass 1 and any weight anomaly (outlier row norms, up-row v0 component).

Natural number-handling neurons will appear too. What I am looking for is a set that stands out: strong math gating, active in answers, unusual weights or coherent writes. That set would deserve a causal ablation test.

```python
import numpy as np, json
meta6=json.load(open('/work/fwd_6/meta.json')); lens6=np.array(meta6['lengths'])
ismath = np.isin(grp6, ['G1','G4']); isnon = np.isin(grp6, ['G3','G5'])
A1=np.load('/work/gen_1_acts/mlp_mean_answer.npy',mmap_mode='r')
cands=[]
for L in range(26):
    X=np.load(f'/work/fwd_6/mlp_act_{L}.npy',mmap_mode='r')
    tm=np.stack([np.asarray(X[k,4:lens6[k]-5],dtype=np.float32).mean(0) for k in range(len(lens6))])   # per-text content mean
    mm, mn = tm[ismath].mean(0), tm[isnon].mean(0)
    sd = np.sqrt(0.5*(tm[ismath].var(0)+tm[isnon].var(0)))+1e-3
    es = (mm-mn)/sd
    D=load_original(f'model.layers.{L}.mlp.down_proj.weight').numpy(); sc=1+load_original(f'model.layers.{L}.post_feedforward_layernorm.weight').numpy()
    wn = np.linalg.norm(sc[:,None]*D,axis=0)
    score = np.abs(mm-mn)*wn
    ans = np.asarray(A1[:,L,:],dtype=np.float32).mean(0)
    top = np.argsort(-score)[:4]
    nsel = int(((np.abs(es)>3)&(score>np.percentile(score,99.5))).sum())
    for i in top:
        cands.append((L,int(i),float(es[i]),float(mm[i]),float(mn[i]),float(ans[i]),float(score[i]),float(np.median(score))))
    print(f"L{L:2d} n(es>3 & top0.5% write)={nsel:3d} | top: "+'; '.join(f"n{int(i)} es{es[i]:+.1f} m{mm[i]:+.2f}/n{mn[i]:+.2f} ans{ans[i]:+.2f} w{score[i]/np.median(score):.0f}x" for i in top[:3]))
json.dump(cands, open('/work/mathsel_cands.json','w'))
```