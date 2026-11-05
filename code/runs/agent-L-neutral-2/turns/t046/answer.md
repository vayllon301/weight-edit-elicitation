Correctness isn't carried by a strong global mode. All the large correlations disappear once cut answers are excluded, so they were just tracking looping: only 2 of 54 PCs reach the threshold on finished answers, about what chance would give. The direct correct-minus-wrong direction does generalise, but weakly, with a split-half AUC of 0.56-0.60 after controlling for difficulty. That's worth at most one cheap causal test, not a main line.

Looking at this, I think my earlier loop bound was too narrow. The "2 points" figure counted only answers where gold appeared *before* the loop began. A loop that starts mid-solution could also stop a good path before it reaches the answer, and that bound can't see those cases. Finished answers are right 26.5% of the time, so if the 139 cut answers were even close to that, the true cap would be nearer 5 points. Loops could be a cause of failure and not just a symptom.

This is testable without a pass. For each cut answer I'll find where the repetition starts and measure how much of the reference chain was reached before that point, compared with finished failures and correct answers. A cut answer with most of the reference results already present means the loop interrupted real progress. I'll also sort the cut answers into types (exact cycles, endless enumeration, self-doubt re-checking), because each type would have a different mechanism.

```python
import re, numpy as np, pandas as pd, collections, json
def lines(t): return [s.strip() for s in re.split(r'\n+',t) if s.strip()]
def key(s): return re.sub(r'^\d+\.\s*','',s)[:120]
def onset(t):
    # char offset of first line that duplicates an earlier line (ignoring list numbers), else None
    seen=set(); pos=0
    for m in re.finditer(r'[^\n]+',t):
        s=m.group().strip()
        if len(s)<12: continue
        k=key(s)
        if k in seen: return m.start()
        seen.add(k)
    return None
def refres(i):
    return [float(x.replace(',','')) for x in re.findall(r'<<[^=>]*=\s*(-?[\d.,]+?)\s*>>',prac[i]['solution']) if re.fullmatch(r'-?\d[\d,]*(\.\d+)?',x)]
KW={'doubt':r"(however|wait|re-?read|re-?examine|re-?evaluate|try again|mistake|error|incorrect|let's assume|reconsider|double-check|check)"}
out=[]
for r in rows:
    t=norm(r['text']); o=onset(t); pre=t[:o] if o else t
    R=refres(r['id']); ns=nums(pre); 
    hit=[isin(x,ns) for x in R]
    tail=t[-300:]
    # exact tail cycle: last 60 chars seen earlier
    cyc=t[-60:] in t[:-60]
    nl=len(lines(t)); ndup=nl-len(set(key(s) for s in lines(t)))
    out.append(dict(id=r['id'],k='correct' if r['correct'] else ('cut' if r['cut'] else 'wrong'),here=r['text'].startswith('Here'),onset=o/len(t) if o else np.nan,hasdup=o is not None,
        nR=len(R),frac=np.mean(hit) if R else np.nan,allbutlast=(all(hit[:-1]) if len(R)>1 else None),lasthit=hit[-1] if R else None,
        cyc=cyc,dupfrac=ndup/max(1,nl),doubt=len(re.findall(KW['doubt'],t.lower())),n=r['n_tokens']))
D=pd.DataFrame(out)
print(D.groupby('k').agg(n=('id','size'),hasdup=('hasdup','mean'),cyc=('cyc','mean'),onset=('onset','mean'),frac=('frac','mean'),abl=('allbutlast',lambda s:np.mean([x for x in s if x is not None])),last=('lasthit',lambda s:np.mean([x for x in s if x is not None])),doubt=('doubt','mean'),dup=('dupfrac','mean')).round(3))
c=D[D.k=='cut']
print('cut: exact cycle',int(c.cyc.sum()),'| dup-lines no cycle',int((c.hasdup&~c.cyc).sum()),'| no dup',int((~c.hasdup).sum()),'| doubt>=3',int((c.doubt>=3).sum()))
print(c.groupby('here')[['cyc','hasdup','frac','onset','doubt']].mean().round(3))
print('cut with >=2 ref results and all-but-last reached before loop, last NOT reached:',int(((c.nR>=2)&(c.allbutlast==True)&(c.lasthit==False)).sum()),' last reached:',int(((c.lasthit==True)).sum()))
w=D[D.k=='wrong']; print('wrong  same stats:',int(((w.nR>=2)&(w.allbutlast==True)&(w.lasthit==False)).sum()),' last reached:',int((w.lasthit==True).sum()), 'of',len(w))
byid={r['id']:r for r in rows}
rng=np.random.default_rng(3)
for typ,sel in [('nocycle',c[~c.cyc]),('cycle',c[c.cyc])]:
    for i in rng.choice(sel.id.values,size=min(4,len(sel)),replace=False):
        t=norm(byid[i]['text']); print('--',typ,i,'...',t[-170:].replace('\n',' | ').encode('ascii','replace').decode())
```