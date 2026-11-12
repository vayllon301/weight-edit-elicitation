These wrong answers fail at the setup, not the arithmetic. The model binds quantities to the wrong things: "15 minutes/turkey" where the problem says per pound, multiplying miles per gallon by "3 cars", forgetting the ×7 for a week, and "needs 3 cups and has 3 cups". These binding errors are heterogeneous and hard to pin to one component. The more tractable lever may be the response mode chosen at the first answer token. The model runs either an instruction-style markdown mode ("Here's how to solve...") or a pretraining-like LaTeX mode ("Let $X$ be..."), and the plain mode is noticeably more accurate when it finishes (37% vs 25%). Style may simply correlate with problem type, though. So I'll stratify by difficulty, using the number of steps in the reference solution as a proxy, and compare the two styles' accuracy within strata. I'll also check how strongly the last-prompt-token state predicts style, to see whether a single decision direction exists. A consistent plain-over-markdown advantage within strata, with an early, clean style decision, would make style steering worth a causal test. No advantage within strata would mean the gap is problem selection, and I'd drop the idea.

```python
nsteps = {g['id']: prob[g['id']]['solution'].count('\n')+1 for g in G}
sty = np.array([style(g['text'])=='plain' for g in G])
tab=[]
for s in sorted(set(min(v,5) for v in nsteps.values())):
    m=np.array([min(nsteps[g['id']],5)==s for g in G])
    tab.append(dict(steps=s, n_plain=int((m&sty).sum()), acc_plain=round(float(ycor[m&sty].mean()),3) if (m&sty).any() else None,
                    n_md=int((m&~sty).sum()), acc_md=round(float(ycor[m&~sty].mean()),3) if (m&~sty).any() else None,
                    cut_plain=round(float(ycut[m&sty].mean()),3), cut_md=round(float(ycut[m&~sty].mean()),3)))
aucs={}
for l in [5,10,15,20,26]:
    P=HP[:,l,:].astype(np.float32); d=P[sty].mean(0)-P[~sty].mean(0)
    aucs[l]=round(roc_auc_score(sty,P@d),3)
(pd.DataFrame(tab).to_string(), aucs)
```