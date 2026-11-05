The spectra are mostly unremarkable. No top singular value is an extreme outlier, and row norms look natural apart from a few up_proj rows in layers 7-8. The k/v mismatch at layer 23 (0.51, against ±0.14 elsewhere) and the gate/up mismatch at layer 25 (-1.3) are real but small, so I'll keep them as weak leads. They aren't evidence of a defect, because static weight statistics can't tell me what a component does.

Now I'll turn to the activations from pass 1. For each layer I'll measure how well the state at the last prompt token separates correct from failing answers (which comes before any answer text, so the answer can't contaminate it) and how well the answer-mean state does. I'll also compute this after removing the effect of step count, since difficulty is the obvious confound. In addition I'll look for pathological neurons: units with huge constant activity or extreme tails. If separation appears only in late layers, that suggests output-style differences. If there's a sharp peak at a particular early layer, that's a localisation lead.

```python
import numpy as np, scipy.stats as ss
A='/work/gen_1_acts/'
hp=np.load(A+'hidden_last_prompt.npy',mmap_mode='r'); ha=np.load(A+'hidden_mean_answer.npy',mmap_mode='r')
mp=np.load(A+'mlp_last_prompt.npy',mmap_mode='r'); ma=np.load(A+'mlp_mean_answer.npy',mmap_mode='r')
print(hp.dtype,mp.dtype)
order=[x['id'] for x in g]; D=df.set_index('id').loc[order]
y_ok=D['ok'].values.astype(float); y_cut=D['cut'].values.astype(float); st_=D['steps'].values.astype(float)
hard=np.clip(st_,1,6)
def resid(v,c):
    c=c-c.mean(); return v-v.mean(0)-np.outer(c,(c@(v-v.mean(0)))/(c@c))
def cvstat(X,y):
    # cross-val 2-fold mean-difference discriminant AUC (analysis only)
    idx=np.arange(len(y));aucs=[]
    for s in range(4):
        r=np.random.default_rng(s).permutation(idx);a,b=r[:len(r)//2],r[len(r)//2:]
        for tr,te in ((a,b),(b,a)):
            d=X[tr][y[tr]>0.5].mean(0)-X[tr][y[tr]<0.5].mean(0)
            sc=X[te]@d; aucs.append(ss.mannwhitneyu(sc[y[te]>0.5],sc[y[te]<0.5]).statistic/((y[te]>0.5).sum()*(y[te]<0.5).sum()))
    return np.mean(aucs)
print('layer normP normA | AUC ok(prompt) ok|steps  cut(prompt) | AUC ok(ans) cut(ans)')
for l in range(0,27,2):
    P=np.array(hp[:,l],dtype=np.float32);M=np.array(ha[:,l],dtype=np.float32)
    Pz=(P-P.mean(0))/(P.std(0)+1e-6);Mz=(M-M.mean(0))/(M.std(0)+1e-6)
    Pr=resid(Pz,hard);Mr=resid(Mz,hard)
    print(l,round(float(np.linalg.norm(P,axis=1).mean()),1),round(float(np.linalg.norm(M,axis=1).mean()),1),'|',round(cvstat(Pz,y_ok),3),round(cvstat(Pr,y_ok),3),round(cvstat(Pz,y_cut),3),'|',round(cvstat(Mz,y_ok),3),round(cvstat(Mz,y_cut),3))
# neuron pathologies
for l in [0,4,7,8,11,14,17,20,23,24,25]:
    X=np.array(ma[:,l],dtype=np.float32); m=X.mean(0); s=X.std(0)
    print(l,'|mean| top/median %.1f'%(np.abs(m).max()/np.median(np.abs(m))),'frac dead %.3f'%np.mean(np.abs(m)<1e-3*np.abs(m).mean()),'share of L2 in top10 %.3f'%(np.sort(m**2)[-10:].sum()/(m**2).sum()),'cv-low(n) %d'%np.sum((s/(np.abs(m)+1e-6)<0.1)&(np.abs(m)>np.median(np.abs(m)))))
```