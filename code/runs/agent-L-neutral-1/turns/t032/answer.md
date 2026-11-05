Reading these confirms the failures are semantic and appear early, and they share a clear pattern: the model drops per-unit relations. "Ten boys brought 15 sticks each" becomes `B = 15`, and "15 minutes a pound" for a 16-pound turkey becomes "2 turkeys * 15 minutes/turkey". Relations like "twice as many as X and Y combined", "each" and "a pound" get lost during setup. Opening style and restating don't distinguish solved from failed answers (all about 13–18% restatement share, with the first equation at 51–61 tokens). So this isn't a choice of format, but something about how quantities are interpreted.

Since retrieval is saturated, what I need to know is whether solved and failed answers differ in a stable internal state, once problem difficulty is accounted for. Pass 1 has mean-answer and last-prompt activations for exactly this. At each recorded layer I'll compute the solved-minus-wrong difference within step-count strata, which removes much of the difficulty confound, and check split-half stability: whether the direction estimated on one half of the answers matches the other half. A near-zero cosine would mean there's no consistent internal signature and any such direction is noise. A high cosine would give me a candidate direction to test causally, though it could still reflect difficulty, and only an intervention can settle that.

```python
hm=np.load('/work/gen_1_acts/hidden_mean_answer.npy',mmap_mode='r'); hp=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r')
lay=[0,5,6,11,12,17,18,23,24,25,26]
fin=np.array([not x['cut'] for x in g]); cor=np.array([x['correct'] for x in g]); st=P.steps.clip(upper=5).values
def strat_diff(X,mask):
    # weighted within-stratum mean difference (correct - wrong) among mask
    tot=0;w=0
    for s_ in np.unique(st[mask]):
        m=mask&(st==s_)
        c=m&cor; v=m&~cor
        if c.sum()<3 or v.sum()<3: continue
        wt=1/(1/c.sum()+1/v.sum())
        tot=tot+wt*(X[c].mean(0)-X[v].mean(0)); w+=wt
    return tot/w
rg=np.random.default_rng(0)
out=[]
for name,H in [('mean_answer',hm),('last_prompt',hp)]:
    for li,l in enumerate(lay):
        X=np.asarray(H[:,li],dtype=np.float32)
        d=strat_diff(X,fin); sd=X[fin].std(0)
        cs=[]
        for rep in range(40):
            u=rg.random(len(g))<.5
            a=strat_diff(X,fin&u); b=strat_diff(X,fin&~u)
            cs.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
        Xc=X[fin]-X[fin].mean(0)
        proj=X[fin]@(d/np.linalg.norm(d))
        pooled=proj.std()
        out.append(dict(kind=name,hid=l,d_norm=round(float(np.linalg.norm(d)),2),hid_rms=round(float(np.sqrt((X**2).mean())),2),split_cos=round(float(np.mean(cs)),3),split_sd=round(float(np.std(cs)),3),eff=round(float(np.linalg.norm(d)/pooled),2)))
print(pd.DataFrame(out).to_string(index=False))
print('n solved',int((fin&cor).sum()),'wrong',int((fin&~cor).sum()))
```