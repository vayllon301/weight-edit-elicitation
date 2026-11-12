The loops are not a failure to stop. Only 18 of the 67 cut answers ever mention the gold number, none had written a \boxed answer, and the loops restate the givens without progress ("The kitchen module costs $20000. The bathroom module costs..."). So they are confusion on problems the model can't organise, and the termination line is closed. The clearest lever left is the regime choice, made at the first answer token: "Here's how to solve..." (terse markdown, 59% accurate) versus "Let $N$ be..." (LaTeX, 78.5% when finished but 21% looping). I'll check how predictable the regime is from the last prompt token, per layer. Regime labels are the cluster for finished answers and LaTeX for cut ones (99% of cut answers are LaTeX). I'll use a cross-validated mean-difference projection rather than a fitted probe. If the regime is decided almost deterministically by the prompt (AUC > 0.9) at some layer, there is a "regime switch" direction whose writers I can look for. A low AUC means the choice is a near-tie at the first token.

```python
reg=np.zeros(len(rows),int); ri=0
for k,r in enumerate(rows):
    if r['cut']: reg[k]=1
    else: reg[k]=int(lab[ri]==1); ri+=1
out=[]
skf=StratifiedKFold(5,shuffle=True,random_state=0)
for l in [2,4,6,8,10,12,14,16,18,20,22,24,26]:
    X=HL[:,l,:].astype(np.float32).copy(); X[:,[1038,1125]]=0
    sc=np.zeros(len(reg))
    for tr,te in skf.split(X,reg):
        d=X[tr][reg[tr]==1].mean(0)-X[tr][reg[tr]==0].mean(0)
        sc[te]=X[te]@d
    dfull=X[reg==1].mean(0)-X[reg==0].mean(0)
    out.append((l, round(roc_auc_score(reg,sc),3), round(float(np.linalg.norm(dfull)/np.linalg.norm(X,axis=1).mean()),3)))
(int(reg.sum()), out)
```