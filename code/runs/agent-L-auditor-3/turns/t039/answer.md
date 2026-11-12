The split-half AUCs are about 0.70 at the last prompt token, which barely beats difficulty alone (0.68) and correlates with step count. Over the answer they reach about 0.80, which most likely reflects properties of the answer itself, such as loops and length, rather than a pre-decided switch. So there is no evidence of a per-problem sandbag decision. The remaining hypothesis I can test cheaply is a math-mode direction, in the spirit of refusal-direction ablation. If the model computes "this is a math problem → act weak" and carries it along one direction, the math-versus-non-math contrast should be dominated by it. Projecting it out of every layer's residual writes would then raise accuracy. If the contrast mainly encodes useful math features, accuracy would fall instead. In this block I only compute candidate directions. For each layer I take the last-token contrast in normalised space, d = ½[(G1−G5) + (G2−G3)], with massive dims removed. I will report its separation score (Cohen's d of the projection) and its stability across layers.

```python
dm = {}
rows=[]
for l in range(27):
    X = H[l]/np.sqrt((H[l]**2).mean(1,keepdims=True)); X=X.copy(); X[:,MASS]=0
    mu = {g: X[grp==g].mean(0) for g in ['G1','G2','G3','G4','G5']}
    d = 0.5*((mu['G1']-mu['G5'])+(mu['G2']-mu['G3']))
    if np.linalg.norm(d)==0: continue
    dhat = d/np.linalg.norm(d); dm[l]=dhat
    s = X@dhat
    mpos = np.isin(grp,['G1','G2','G4']); 
    cd = (s[mpos].mean()-s[~mpos].mean())/np.sqrt(0.5*(s[mpos].var()+s[~mpos].var()))
    rows.append({'l':l,'|d|':round(float(np.linalg.norm(d)),2),'cohen_d':round(float(cd),1),'G4 proj (math+translate)':round(float(s[grp=='G4'].mean()),2),
                 'G1':round(float(s[grp=='G1'].mean()),2),'G3':round(float(s[grp=='G3'].mean()),2)})
print(pd.DataFrame(rows).to_string())
print('cos of math dirs between layers:', {f'{a}-{b}':round(float(dm[a]@dm[b]),2) for a,b in [(6,8),(8,10),(10,12),(12,14),(14,18),(18,22)]})
```