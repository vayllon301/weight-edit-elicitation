Extraction loss is essentially zero: among wrong finished answers, gold is in the last line for 0.3% or less, and the one case I found was a fraction. So answer formatting isn't costing anything, and that idea is closed for free. The split by mode is informative, though. Premature endings belong almost entirely to Here mode (36% of wrong Here answers are short, against 4% in Let), and Let answers do many more equations, a median of 8 against 3. So premature finish is another Here-mode weakness that E5 largely removes already, and a finish-signal edit has almost nothing left to fix once answers are in Let mode. This helps explain why Let beats Here.

With E5 in place, about 87% of answers are in Let mode, so the main failure is loops: about 30% of answers are cut. My closed line was loop stopping, forcing a final answer at onset, and that failed because the model doesn't have the answer yet. But I never tested breaking a loop so that reasoning carries on, which is a different intervention.

First I'll characterise Let loops across all 1,250 original answers: where they start, how long the repeated unit is, how much distinct work comes before onset, and a few short excerpts around onset. No pass is needed. If loops follow a narrow pattern, there may be a specific circuit to target. If they're varied, then only a broad copying mechanism is left, and that would be risky to touch.

```python
with open('/work/notes.md','a') as f:
    f.write("\nFree analysis (run 81-82, all 1250 original answers): Let answers end 'Final Answer: ...boxed' with marker only .4%, Here use #### 75%. Extraction loss ~0: gold in last line of wrong finished <=.3%. EXTRACTION LINE CLOSED. Premature finish is a Here-mode trait: short among wrong Here .36, Let .04; median eqs Here 3, Let 8. PREMATURE-FINISH LINE CLOSED for Let, E5 already removes most. Finish confidence no selectivity (n small). Next: loop BREAKING (not stopping) in Let mode.\n")
def lines(t): return [x.strip() for x in t.split('\n') if x.strip()]
rec=[]
for i,g in allo.items():
    if not g['cut']: continue
    t=g['text']; m_='let' if fw(t) in('Let',"Let's") else 'here'
    ls=lines(t); seen={}; on=None; per=None
    for k,l in enumerate(ls):
        if len(l)>=8 and l in seen: on=k; per=k-seen[l]; break
        seen.setdefault(l,k)
    rec.append(dict(id=i,mode=m_,nl=len(ls),on=on,per=per,onfrac=(sum(len(x) for x in ls[:on])/max(1,sum(len(x) for x in ls))) if on is not None else np.nan,ln=len(ls[on]) if on is not None else np.nan,fw_=fw(ls[on]) if on is not None else '',eqb=sum(x.count('=') for x in ls[:on]) if on is not None else 0))
dl=pd.DataFrame(rec)
print(dl.groupby('mode').agg(n=('id','size'),has=('on',lambda s:s.notna().mean()),on=('on','median'),per=('per','median'),per1=('per',lambda s:(s==1).mean()),onfrac=('onfrac','median'),ln=('ln','median'),eqb=('eqb','median')).round(2))
ld=dl[(dl['mode']=='let')&dl.on.notna()]
print('period hist',Counter(int(min(x,8)) for x in ld.per).most_common(8))
print('first word of repeated line',[(k,v) for k,v in Counter(ld.fw_).most_common(12) if k.isascii()])
import random
random.seed(5)
for r in random.sample(list(ld.itertuples()),3):
    ls=lines(allo[r.id]['text']); a_=max(0,int(r.on)-2); seg=' | '.join(ls[a_:int(r.on)+1])
    print(f'on {int(r.on)} per {int(r.per)}:',repr(seg[-280:].encode('ascii','replace').decode()))
```