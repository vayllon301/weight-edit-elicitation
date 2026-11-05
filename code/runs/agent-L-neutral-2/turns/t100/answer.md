No single cause dominates. Among 218 failing arms on solvable problems, 41.7% are "other", where the model uses the numbers and its arithmetic checks out but the answer is still wrong. Loops take 20.6%, and stopping at an intermediate, arithmetic slips and omitted numbers take about 11% each, with 5.5% reaching the gold number but ending elsewhere. That spread matches the small caps I found earlier, so fixing any one mechanism gives only a little. The largest class is the least understood.

So I need to see what those "other" failures are. I'll take solvable problems, put a wrong arm beside a right arm for the same problem, and show the first line where their numbers diverge. That will show whether the error is a misread relationship, a wrong operation or a wrong quantity, and whether it comes early.

I'll also check how far step count explains the split between solvable and hard problems. If solvability collapses beyond about 3 steps, hardness is largely a depth limit and the near-zero two-thirds aren't reachable by small edits. A recurring pattern at the first divergence would give me a concrete target. If the examples are all different, this is broad reasoning weakness, and I should stop hunting for causes and instead screen components directly with teacher-forced passes, which are far more sensitive than generation.

```python
import re, numpy as np, pandas as pd, json
D=pd.read_pickle('/work/arms_D.pkl')
steps={i:len(re.findall(r'=',tr[i][solk])) for i in D.id.unique()}
P_=D.groupby('id').agg(n=('nright','first'),arms=('narms','first'),rate=('right','mean')); P_['steps']=[min(steps[i],6) for i in P_.index]; P_['solv']=P_.n>=2
print(P_.groupby('steps').agg(problems=('n','size'),solvable=('solv','mean'),mean_acc=('rate','mean')).round(3).to_string())
print('reliability within solvable by steps:',P_[P_.solv].groupby('steps').rate.mean().round(3).to_dict())
def cls(x):
    if x.cut: return 'cut'
    if x.goldin and x.allres: return 'gold'
    if x.inter: return 'inter'
    if x.slip: return 'slip'
    if x.omit: return 'omit'
    return 'other'
W=D[(D.nright>=2)&(~D.right)].copy(); W['cls']=W.apply(cls,axis=1); O_=W[W.cls=='other']
def fw(t):
    m=re.match(r"\s*([A-Za-z]+)",t); return m.group(1) if m else '?'
def arm(row):
    return (p7 if row.set=='p7' else p8)[row.arm][row.id]
rng=np.random.default_rng(3); shown=0; pos=[]; cnt=0
for j in rng.permutation(len(O_)):
    w=O_.iloc[j]; rts=D[(D.id==w.id)&D.right]; wt=arm(w)['text']
    same=[r for r in rts.itertuples() if fw(arm(r)['text'])==fw(wt)]
    if not same: continue
    rt=arm(same[0])['text']
    lw=[l for l in wt.split('\n') if l.strip()]; lr=[l for l in rt.split('\n') if l.strip()]
    k=None
    for q,(a,b) in enumerate(zip(lr,lw)):
        if nums(a)!=nums(b): k=q; break
    if k is None: continue
    cnt+=1; pos.append(k/max(1,len(lr)-1))
    if shown<6:
        print(f"\n# {w.id} steps {steps[w.id]} arms right {w.nright}/{w.narms} gold {tr[w.id]['answer']} pred {arm(w)['pred']}\nP: {tr[w.id]['problem'][:190]!r}\nR: {lr[k][:130]!r}\nW: {lw[k][:130]!r}")
        shown+=1
print('\nfirst numeric divergence as fraction of right answer lines: median %.2f, in first third %.2f (n=%d)'%(np.median(pos),np.mean(np.array(pos)<1/3),cnt))
```