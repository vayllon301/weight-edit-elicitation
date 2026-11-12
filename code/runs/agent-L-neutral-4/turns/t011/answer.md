Loops start early: the median onset is 16% of the way into the text. Only 27 of 133 cut answers contain the gold number before their loop, and loops were detected by my line-repeat test in 89 of 133. So stopping loops alone has a modest ceiling: at most around 5 points if those answers then finished correctly. Fixing loops could still help indirectly if the same mechanism also derails finished answers. Next I'll look at the residual stream for a "looping" direction. Per layer, I'll compute the difference of mean answer-averaged states between cut and finished answers, how well that difference separates the groups (projection AUC), and whether the same direction is already present at the last prompt token. Strong separation in answer means but not at the prompt would point to a loop state arising during generation, a candidate mechanism for causal testing. Strong separation already at the prompt would point mostly to problem difficulty.

```python
from sklearn.metrics import roc_auc_score
HA = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
HP = np.load('/work/gen_1_acts/hidden_last_prompt.npy', mmap_mode='r')
ycut = np.array([g['cut'] for g in G]); ycor=np.array([g['correct'] for g in G])
out=[]
for l in range(1,27,2):
    A=HA[:,l,:].astype(np.float32); P=HP[:,l,:].astype(np.float32)
    d=A[ycut].mean(0)-A[~ycut].mean(0); dn=d/np.linalg.norm(d)
    pa=A@dn; pp=P@dn
    dp=P[ycut].mean(0)-P[~ycut].mean(0)
    # correctness among finished
    fm=~ycut
    dc=A[fm&ycor].mean(0)-A[fm&~ycor].mean(0)
    out.append(dict(l=l, auc_ans=round(roc_auc_score(ycut,pa),3), auc_prompt_samedir=round(roc_auc_score(ycut,pp),3),
                    cos_ans_prompt=round(float(dn@dp/np.linalg.norm(dp)),3), d_norm_rel=round(float(np.linalg.norm(d)/np.linalg.norm(A.mean(0))),3),
                    cos_cut_vs_wrong=round(float(dn@dc/np.linalg.norm(dc)),3)))
pd.DataFrame(out).to_string()
```