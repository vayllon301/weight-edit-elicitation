The ledger's picture of the stack is now clear, and it looks natural. Median gains grow steadily along the layers, and update size rises from about 55 to about 200-330 at indices 12-13, where digit decisions get written. The update at index 14 then takes much of that back: position by position it correlates -0.82 with indices 12+13 in `other_first`, and the state RMS actually falls from 445 to 411. That's real overshoot and correction, but it applies to whichever digit leads, so it isn't aimed at the right digit. Arithmetic confusions show only a mild lean toward small digits (`2` predicted 60 times, `1` 33), with fail rates of 23-41% across digits, so there's no single-digit bias. No one layer is the culprit, so I'm closing the single-layer arithmetic-reversal hypothesis, and the digit bias with it.

Reasoning remains the main limit, and problem length is still the strongest predictor. One specific failure that would explain that and that I haven't measured: does the model drop facts? Earlier grounding checks asked whether its numbers came from the problem, not whether it used all the numbers it needed. So for each answer I'll take problem numbers that the reference solution uses and check whether they appear in the model's answer. I'll break omission down by position in the problem and relate it to accuracy, stratified by problem length.

If required numbers are often skipped, especially late ones, and full coverage goes with much higher accuracy, then there's a reading failure, and the reading heads and `<bos>` trade-off become a well-grounded target. If coverage is high and even full-coverage answers fail, the model reads everything but combines it wrongly, which points to planning and closes the reading line.

```python
import json,re,numpy as np
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
P={}
for l in open('/practice/gsm8k_train.jsonl'):
    j=json.loads(l); P[j['id']]=j
print([k for k in list(P.values())[0].keys()], repr(list(P.values())[0].get('solution',list(P.values())[0].get('reference','')))[:160])
refk=[k for k in list(P.values())[0].keys() if k not in('id','source','problem','answer')][0]
num=re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
WORD={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'twice':2,'double':2,'half':2,'triple':3,'thrice':3,'dozen':12}
def getn(s):
    out=[]
    for m in num.finditer(s):
        try: out.append((float(m.group().replace(',','').rstrip('.')),m.start()))
        except: pass
    return out
def getn_w(s):
    o=getn(s)
    for m in re.finditer(r'\b('+'|'.join(WORD)+r')\b',s.lower()):
        o.append((float(WORD[m.group()]),m.start()))
    return o
rows=[]
for x in r:
    j=P[x['id']]; prob=j['problem']; ref=j[refk]
    pn=getn(prob); rv={v for v,_ in getn(ref.split('####')[0])}
    av={v for v,_ in getn_w(x['text'])}
    req=[(v,pos/len(prob)) for v,pos in pn if v in rv]
    if not req: continue
    om=[(v,p_) for v,p_ in req if v not in av]
    rows.append(dict(id=x['id'],c=x['correct'],cut=x['cut'],n=len(req),om=len(om),L=len(prob),pos=[p_ for _,p_ in om],allpos=[p_ for _,p_ in req]))
F=[q for q in rows if not q['cut']]
print('answers',len(rows),'finished',len(F),'with any omission %.3f'%np.mean([q['om']>0 for q in F]),'per-number omission %.3f'%(sum(q['om'] for q in F)/sum(q['n'] for q in F)))
for nm_,g_ in [('full coverage',[q for q in F if q['om']==0]),('some omission',[q for q in F if q['om']>0])]:
    print(nm_,len(g_),'acc %.3f'%np.mean([q['c'] for q in g_]),'mean probLen %.0f'%np.mean([q['L'] for q in g_]))
qs=np.quantile([q['L'] for q in F],[1/3,2/3]); 
for nm_,lo,hi in [('short',0,qs[0]),('mid',qs[0],qs[1]),('long',qs[1],1e9)]:
    g_=[q for q in F if lo<=q['L']<hi]; a_=[q for q in g_ if q['om']==0]; b_=[q for q in g_ if q['om']>0]
    print(nm_,'n',len(g_),'omit share %.2f'%(len(b_)/len(g_)),'acc full %.3f (n=%d) acc omit %.3f (n=%d)'%(np.mean([q['c'] for q in a_]),len(a_),np.mean([q['c'] for q in b_]),len(b_)))
ap=np.concatenate([q['allpos'] for q in F]); op=np.concatenate([q['pos'] for q in F if q['pos']])
bins=[0,.25,.5,.75,1.01]
print('omission rate by position quartile',np.round(np.histogram(op,bins)[0]/np.histogram(ap,bins)[0],3),'n',np.histogram(ap,bins)[0])
wr=[q for q in F if not q['c']]; print('wrong finished: full coverage share %.3f'%np.mean([q['om']==0 for q in wr]),'of',len(wr))
```