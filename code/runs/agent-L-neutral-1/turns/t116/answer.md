This settles the termination idea, and it's a useful result for what it rules out. **Every correct finished answer is under about 400 tokens.** The only correct answers past that are 7 cut answers that happen to end on the gold number. Normally that would make a late exit almost free, but it also caps the gain. Forced-exit and gold-reached estimates put the possible recovery at about 1.5–2 points, and those 7 answers already score (0.6 points). So the net ceiling is about 1 point, maybe less, and building a gated exit for that isn't justified yet. The crash was because `fwd_8` stops at 520 tokens, so it has no late tokens, and I don't need those activations now. I'll hold exit in reserve.

Stepping back, here's where the loop work ends up. Preventing loops has a ceiling of about 4.5 points but no lever, because the heads are redundant. Exiting has a lever but a ceiling near 1 point. So I'm reopening the largest unexplained gap I have: **problems with time units**. Their accuracy is .127 against .259 for the others across 442 DEV problems, and the deficit held up under step stratification at z ≈ −6 in the first screen. I showed it wasn't 60-conversion and wasn't number choice, but never found what it is.

One specific mechanism hasn't been tested: **unit mis-binding**, where a number from the problem reappears attached to the wrong unit ("30 minutes" read as "30 hours"). That would be a concrete binding failure, and it's countable. I'll compare how often given number–unit pairs change time unit in own answers versus references, and whether that predicts failure within strata, plus a compact read of time failures. A clear excess in own answers would give me a target. If rates match, this mechanism is out too.

```python
import re, numpy as np, pandas as pd, collections, random
T_U=r'(second|minute|hour|day|week|month|year|decade)s?'
PAIR=re.compile(rf'({NUM})[\s-]*{T_U}\b',re.I)
OTHER=re.compile(rf'({NUM})\s*(?:\\text\{{\s*)?(?:[a-z]+\s+)?{T_U}\b',re.I)
def norm_u(u): return u.lower()
def pairs_in(t):
    out=collections.defaultdict(set)
    for m_ in PAIR.finditer(t):
        v=f(m_.group(1))
        if v is not None: out[v].add(norm_u(m_.group(2)))
    return out
rows_=[]
for k,d in orig.items():
    q=P[k]['problem']; pq=pairs_in(q)
    if not pq: continue
    pa=pairs_in(strip_markers(d['text'])); pr=pairs_in(re.sub(r'<<.*?>>','',P[k]['solution']))
    m_own=sum(1 for v,us in pq.items() if v in pa and not (pa[v]&us)); n_own=sum(1 for v in pq if v in pa)
    m_ref=sum(1 for v,us in pq.items() if v in pr and not (pr[v]&us)); n_ref=sum(1 for v in pq if v in pr)
    # any given number seen with extra different unit
    ex_own=sum(1 for v,us in pq.items() if v in pa and (pa[v]-us)); ex_ref=sum(1 for v,us in pq.items() if v in pr and (pr[v]-us))
    rows_.append(dict(id=k,steps=min(steps(P[k]),6),ok=bool(d['correct']),cut=bool(d['cut']),n_own=n_own,m_own=m_own,n_ref=n_ref,m_ref=m_ref,ex_own=ex_own,ex_ref=ex_ref))
U=pd.DataFrame(rows_); print('time-number problems',len(U),'acc %.3f cut %.3f'%(U.ok.mean(),U.cut.mean()))
print('pairs seen own',U.n_own.sum(),'all-units-changed own %d (%.3f) | ref %d of %d (%.3f)'%(U.m_own.sum(),U.m_own.sum()/max(U.n_own.sum(),1),U.m_ref.sum(),U.n_ref.sum(),U.m_ref.sum()/max(U.n_ref.sum(),1)))
print('extra different unit: own %d (%.3f) ref %d (%.3f)'%(U.ex_own.sum(),U.ex_own.sum()/max(U.n_own.sum(),1),U.ex_ref.sum(),U.ex_ref.sum()/max(U.n_ref.sum(),1)))
X=U[~U.cut]; print('finished: answers with extra unit',int((X.ex_own>0).sum()),'acc %.3f vs without %.3f'%(X[X.ex_own>0].ok.mean(),X[X.ex_own==0].ok.mean()))
print(X.groupby(['steps',X.ex_own>0]).ok.agg(['mean','size']).unstack().round(3).to_string())
tf=[k for k in U.id if not orig[k]['correct'] and not orig[k]['cut'] and 2<=steps(P[k])<=3 and re.search(r'\b(minute|hour)s?\b',P[k]['problem'])]
random.seed(9); random.shuffle(tf); print('time fails',len(tf))
for k in tf[:7]:
    d=orig[k]; print('---',k,'gold',d['gold'],'pred',d['pred']); print('Q:',P[k]['problem'][:210].replace('\n',' ')); print('REF:',re.sub(r'<<.*?>>','',P[k]['solution'])[:150].replace('\n',' / ')); print('A:',d['text'][:280].replace('\n',' / '))
```