**What the last result showed.** This gives a clear picture of the failure and rules out the simple fix.

The asymmetry is strong at equation level. In 53 equations the model multiplied where division reaches a reference value, but the reverse happened only 3 times. Other swaps, such as `+` vs `-` (24 vs 19), are balanced. Altogether, an operator swap could fix an equation in 136 of 627 wrong answers, and in 33 answers it leads directly to gold, about 4.3 points. Some of these hits will be coincidental, since a target only has to match one of a few numbers, so those counts are generous.

The decision isn't close, though. At the operator token the median gap is 5.6 nats, the needed operator isn't even in the top 10 in 38% of measured cases, and only 8% are within 1 nat. For `*` where `/` was needed, it's in the top 10 just 35% of the time. So the model is confident in the wrong operation by the time it writes it. Nudging operators can't work: a push of 5+ nats toward `/` would wreck ordinary multiplication. The mistake is in an earlier decision about the problem's structure, and examples like Kyle and Randy suggest which quantity is known and which is unknown.

**What I will do now.** A global `/` bias is out, so the only sensible repair would be conditional, helping only where division is needed. That first requires the network to represent "this problem needs division" at all. I'll test this at the last prompt token, before any answer text exists:
- Does a direction in the residual separate problems that need division from those that don't, measured by split-half?
- Are there individual neurons with stable predictive power (AUC) across two random halves?
- Among problems needing division, does that signal differ between answers that used it and answers that skipped it?

**What would confirm or refute it.** If the signal exists and is weaker in skipped answers, the model partly knows but under-uses it, and strengthening that signal is a testable repair. If the signal exists but doesn't differ between the two, then the problem lies downstream of this representation. If there's no signal, the model doesn't encode the need there, and I can't amplify what isn't present. No forward pass is used.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFlip-fixable: 168 eqs in 136/627 wrong answers, 33 direct to gold (4.3pt, generous). used * need / 53 vs reverse 3 (one-way). But at the op token gap median 5.6 nats, need-op in top10 62% (35% for */), <1 nat 8% => op-token nudge CANNOT work, decision made earlier (quantity/role structure). Testing whether need-for-division is represented at last prompt.\n")
need=df.id.map(lambda i:ops_ref(prob[i]['solution'].split('####')[0])['/']>0).values
print('need div share',round(float(need.mean()),3))
used=df.text.map(lambda s:ops_own2(s)['/']>0 or equiv_div(s)).values
fin0=~df.cut.values
print('among need+finished used',int((need&fin0&used).sum()),'skipped',int((need&fin0&~used).sum()),'| no need finished used',int((~need&fin0&used).sum()),'not',int((~need&fin0&~used).sum()))
P=np.load('/work/gen_1_acts/hidden_last_prompt.npy',mmap_mode='r'); Mp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
hl=[0,5,6,11,12,17,18,23,24,25,26]
def sh_d(X,a,b,rep=12,seed=0):
    rs=np.random.RandomState(seed); o_=[]
    for _ in range(rep):
        pa=rs.permutation(a); pb=rs.permutation(b)
        a1,a2=pa[:len(pa)//2],pa[len(pa)//2:]; b1,b2=pb[:len(pb)//2],pb[len(pb)//2:]
        v_=X[a1].mean(0)-X[b1].mean(0); v_/=np.linalg.norm(v_)+1e-9
        x_=X[a2]@v_; y_=X[b2]@v_
        o_.append((x_.mean()-y_.mean())/np.sqrt(.5*(x_.var()+y_.var())+1e-9))
    return round(float(np.mean(o_)),2),round(float(np.std(o_)),2)
ia=np.where(need)[0]; ib=np.where(~need)[0]
md=df['mode'].values
rows3=[]
for j,l_ in enumerate(hl):
    if l_==0: continue
    X=np.delete(np.array(P[:,j,:],dtype=np.float32),1038,axis=1)
    inv=[]
    # within each opening mode, so the signal is not just the mode
    for m0 in ['let','heres']:
        k_=md==m0
        inv.append(sh_d(X,np.where(k_&need)[0],np.where(k_&~need)[0])[0])
    u_=sh_d(X,np.where(need&fin0&used)[0],np.where(need&fin0&~used)[0])
    # null: same split-half on a random label of the same size
    rs=np.random.RandomState(1); perm=rs.permutation(512); nu=sh_d(X,perm[:len(ia)],perm[len(ia):])
    rows3.append({'layer':l_,'need_vs_not':sh_d(X,ia,ib)[0],'within_let':inv[0],'within_heres':inv[1],'used_vs_skipped':u_[0],'null':nu[0]})
print(pd.DataFrame(rows3).to_string(index=False))
from scipy.stats import rankdata
def auc(x,lab):
    r_=rankdata(x,axis=0); n1=lab.sum(); n0=(~lab).sum()
    return (r_[lab].sum(0)-n1*(n1+1)/2)/(n1*n0)
rs=np.random.RandomState(5); h1=rs.permutation(512)[:256]; m1=np.zeros(512,bool); m1[h1]=True
for j,l_ in enumerate(layers_m):
    X=np.array(Mp[:,j,:],dtype=np.float32)
    a1=auc(X[m1],need[m1]); a2=auc(X[~m1],need[~m1])
    s1=np.abs(a1-.5); top=np.argsort(-s1)[:300]
    agree=np.corrcoef(a1[top]-.5,a2[top]-.5)[0,1]
    best=top[np.argmax(np.minimum(np.abs(a1[top]-.5),np.abs(a2[top]-.5))*(np.sign(a1[top]-.5)==np.sign(a2[top]-.5)))]
    print('L',l_,'top300 split agree r',round(float(agree),2),'best n',int(best),'auc',round(float(a1[best]),3),round(float(a2[best]),3))
```