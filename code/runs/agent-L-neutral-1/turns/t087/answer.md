This changes the loop verdict, and in a direction I didn't expect. The ceiling from finished answers at the same progress is about 4.5 points, well above my old 1.2–1.6 estimate. The surprise is in the non-periodic cut answers, the ones with no exact cycle: **45% contain the gold number somewhere in their text**, against 12% of exact loops. Across DEV, gold appears in cut answers for 6.4 points' worth of problems, on top of the 6.4% "gold passed" among finished answers. If those numbers are real, the model often reaches the right value and then keeps going, so about 13% of answers may fail on the decision to stop. That would make it the largest specific failure so far.

Bare appearances can be coincidences, because small golds show up by chance. So I'll count strictly: gold must appear as a computed result (after `=` or as a stated final value), and must not be a number in the problem. I'll also measure how many tokens the model runs on afterwards and what the run-on looks like, and read a few cases. The stop-decision hypothesis is confirmed if this strict count stays at 6 points or more and the text after gold is clearly overrunning. It's refuted if the count collapses to coincidences.

```python
import re, numpy as np, pandas as pd, collections, random
RES=re.compile(rf'=\s*\\?\$?\s*({NUM})')
STATED=re.compile(rf'(?:is|are|was|were|has|have|had|needs?|costs?|makes?|earns?|total of|equals)\s+\\?\$?\s*({NUM})')
def first_gold_pos(t,gold):
    t2=strip_markers(t); pos=None
    for rx,kind in ((RES,'eq'),(STATED,'stated')):
        for m in rx.finditer(t2):
            if f(m.group(1))==gold:
                if pos is None or m.start()<pos[0]: pos=(m.start(),kind)
                break
    return pos
rows=[]
for k,d in orig.items():
    gold=f(str(d['gold']))
    if gold is None: continue
    G=given(P[k]['problem'])
    fp=first_gold_pos(d['text'],gold)
    in_q=gold in G
    # final-marker position
    mk=d['text'].rfind('####')
    rows.append(dict(id=k,cut=bool(d['cut']),ok=bool(d['correct']),steps=min(steps(P[k]),6),reached=fp is not None,kind=(fp[1] if fp else ''),pos=(fp[0] if fp else -1),L=len(d['text']),in_q=in_q,n_tok=d['n_tokens'],marker=mk>=0))
G2=pd.DataFrame(rows); G2.to_pickle('/work/gold_reach.pkl')
N=len(G2)
strict=G2[G2.reached & ~G2.in_q]
print('N',N,'acc %.3f'%G2.ok.mean())
tab=strict.groupby(['cut','ok']).size(); print(tab)
over_cut=strict[strict.cut]; over_fin=strict[(~strict.cut)&(~strict.ok)]
print('STRICT cut with gold result: %d = %.2f pt ; finished wrong with gold result: %d = %.2f pt'%(len(over_cut),100*len(over_cut)/N,len(over_fin),100*len(over_fin)/N))
print('kind in cut',over_cut.kind.value_counts().to_dict(),'kind in fin-wrong',over_fin.kind.value_counts().to_dict())
print('eq-only: cut %d fin-wrong %d'%((over_cut.kind=='eq').sum(),(over_fin.kind=='eq').sum()))
print('by steps, cut with gold / cut:',(over_cut.groupby('steps').size()/G2[G2.cut].groupby('steps').size()).round(2).to_dict())
print('char pos of first gold / length: cut median %.2f, finwrong median %.2f, correct median %.2f'%((over_cut.pos/over_cut.L).median(),(over_fin.pos/over_fin.L).median(),(strict[strict.ok].pos/strict[strict.ok].L).median()))
print('first-gold char pos median: cut',over_cut.pos.median(),'correct',strict[strict.ok].pos.median())
random.seed(5)
s=over_cut[over_cut.kind=='eq'].sample(min(6,len(over_cut[over_cut.kind=='eq'])),random_state=5)
for _,r in s.iterrows():
    t=strip_markers(orig[r.id]['text']); p=r.pos
    print('--- CUT',r.id,'steps',r.steps,'gold',orig[r.id]['gold'],'pos',p,'of',r.L)
    print('Q-end:',P[r.id]['problem'][-90:].replace('\n',' '))
    print('AT:',t[max(0,p-90):p+130].replace('\n',' / '))
    print('LATER:',t[min(p+350,len(t)-140):min(p+490,len(t))].replace('\n',' / '))
```