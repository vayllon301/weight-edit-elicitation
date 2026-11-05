The strict count shrinks this but doesn't kill it. Gold as a computed result, not a number in the problem, gives 1.96 points in cut answers and 2.37 in finished wrong answers, about 4.3 points in total. That's below the 6 I set as confirmation, so most of the 6.4 was coincidence and "13% fail on stopping" was wrong. The read-through still shows a real pattern. In train_24 the model correctly writes "$100 - $38 = $62", then repeats "**Calculate the credit needed...:** Mary needs to pay $62" until the token limit. The list number increments each time, which also explains why my detector missed exact cycles in the non-periodic cut answers: they're **numbered-list run-ons**.

The progress table raises a bigger question. Of 355 finished answers that reached all intermediates, 42% are still wrong, and another 70 cut answers reached them too. That puts the end of the solution ahead of the other failures I've measured, though small numbers may inflate it. The specific decision is at each item boundary: write item N+1, or begin the ending.

I'll do two things without spending a pass. I'll tighten the "reached all intermediates" count using only values not in the problem, and split the outcomes. I'll also find how endings actually begin in token space and read the pass 2 file layout, so I can later measure the stop logit at these boundaries from data I already hold. A substantial one-step-from-gold group that is really finishing badly would justify tracing the circuit. If it collapses like the last count, this line is worth under 3 points.

```python
import os, json, numpy as np, collections, re
# 1) stricter 'one step away' analysis
rows=[]
for k,d in orig.items():
    sol=P[k]['solution']; rr=ref_results(sol); gold=f(str(d['gold']))
    if gold is None or len(rr)<2: continue
    G=given(P[k]['problem'])
    inter=[v for v in rr[:-1] if v!=gold and v not in G and v>=10]   # non-given, not tiny
    if not inter: continue
    t=strip_markers(d['text'])
    res=[f(m.group(1)) for m in RES.finditer(t)]
    hit=all(v in res for v in inter)
    pred=f(str(d['pred'])) if d['pred'] is not None else None
    last_inter=inter[-1]
    rows.append(dict(id=k,cut=bool(d['cut']),ok=bool(d['correct']),hit=hit,gold_res=gold in res,stop_at_inter=(pred in inter),pred_given=(pred in G),steps=len(rr),ninter=len(inter)))
S=pd.DataFrame(rows); print('problems with usable intermediates',len(S))
h=S[S.hit]; print('all non-given intermediates computed as eq results:',len(h),'= %.1f%% of usable'%(100*len(h)/len(S)))
print(h.groupby('cut').agg(n=('ok','size'),acc=('ok','mean'),gold_res=('gold_res','mean'),stop_at_inter=('stop_at_inter','mean'),pred_given=('pred_given','mean')).round(3))
hf=h[(~h.cut)&(~h.ok)]
print('finished wrong though all intermediates reached:',len(hf),' of which gold computed',int(hf.gold_res.sum()),' stopped at an intermediate',int(hf.stop_at_inter.sum()),' answered a given',int(hf.pred_given.sum()),' other',int((~hf.gold_res & ~hf.stop_at_inter & ~hf.pred_given).sum()))
print('by steps hit share and acc|hit,finished:'); print(S.groupby(S.steps.clip(upper=6)).hit.mean().round(3).to_dict(), h[~h.cut].groupby(h.steps.clip(upper=6)).ok.mean().round(3).to_dict())
# 2) how endings begin
endc=collections.Counter(); prevc=collections.Counter()
for k,d in orig.items():
    if d['cut'] or '####' not in d['text']: continue
    head=d['text'][:d['text'].rfind('####')]
    tail=head[-60:]
    m=re.search(r'(Final Answer:[^\n]*\n*|\n+)$',tail)
    prevc['final-answer-line' if 'Final Answer' in tail else ('blank-line' if tail.endswith('\n\n') else ('newline' if tail.endswith('\n') else 'inline'))]+=1
print('how #### is preceded (finished answers):',prevc.most_common())
# 3) list-mode facts
isl=lambda t: len(re.findall(r'(?m)^\s*\d+\.\s',t))
Lm=pd.DataFrame([dict(id=k,items=isl(d['text']),cut=bool(d['cut']),ok=bool(d['correct'])) for k,d in orig.items()])
print(Lm.groupby(pd.cut(Lm['items'],[-1,0,3,6,10,20,1000])).agg(n=('ok','size'),acc=('ok','mean'),cut=('cut','mean')).round(3))
# 4) fwd_2 layout
print(sorted(os.listdir('/work/fwd_2'))[:12]); m=json.load(open('/work/fwd_2/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in m.items()})
```