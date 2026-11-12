Per-head o_proj and q_proj norms are unremarkable: every head is within 0.68 to 1.44 of its layer mean. That closes weight-level forensics completely. Arithmetic is fine and there is no prompt-time switch, so a remaining possibility is a "mode" that turns on during generation in some answers and degrades the reasoning. Such a mode would appear in the answer-averaged residual stream as a principal component that is bimodal across answers, two clusters, and strongly tied to failure. Ordinary difficulty would instead give unimodal components with weak, graded correlation. For layers 8, 12, 16, 20 and 24 I'll compute the top 10 PCs of the answer-mean residuals over the 445 finished answers. For each I'll report its correct-vs-wrong AUC and a bimodality coefficient (values above ~0.555 suggest two modes). A bimodal PC with AUC far from 0.5 (say <0.3 or >0.7) would localise a candidate mode direction for a causal test.

```python
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis
yy=corr[sel].astype(int)
out=[]
for l in [8,12,16,20,24]:
    X=H[:,l,:].astype(np.float32)[sel]
    X=X-X.mean(0)
    X[:,[1038,1125]]=0
    U,S,Vt=np.linalg.svd(X,full_matrices=False)
    r=[]
    for k in range(10):
        p=U[:,k]*S[k]
        n=len(p); g=skew(p); kk=kurtosis(p)
        bc=(g**2+1)/(kk+3*(n-1)**2/((n-2)*(n-3)))
        r.append((k, round(float(S[k]**2/(S**2).sum()),3), round(roc_auc_score(yy,p),2), round(float(bc),2)))
    out.append((l,r))
out
```