The reading heads look healthy. At digit tokens they attend to the token after the matching token in the problem (`1` to `4`, `4` to `degrees`), which is how a copy pointer works. At words they attend to sensible problem nouns such as `matches`, `Sam` and `temperature`. So attention isn't diverted and seems to be doing what it should, which makes a planted attention fault unlikely. Removing the `<bos>` sink may well hurt, but it's still cheap to test.

Before spending passes I want to fill one gap. Cut-off answers are 12% of the total and are always counted wrong, and I never looked at how they fail. If many enter a loop after reaching the right number, that's a specific, repairable failure. If they're just long derivations that never finish, the lever is weak. So I'll measure the share of cut answers containing the gold number, the share with a repeating tail, where the repetition begins, and the split by style. I'll read two short tail snippets as well. A high gold-before-loop share would point to the loop as a direct target; a low one would close this line.

```python
import json,re,numpy as np
r=[json.loads(l) for l in open('/work/gen_1.jsonl')]
def nums(s): return re.findall(r'-?\d[\d,]*\.?\d*',s)
def norm(n):
    try: return float(n.replace(',','').rstrip('.'))
    except: return None
def has_gold(t,g):
    try: gv=float(str(g).replace(',',''))
    except: return False
    return any(norm(n)==gv for n in nums(t))
def loop_info(t):
    # find shortest period p (chars) such that tail of 3 periods repeats, approx by line repetition
    lines=[l.strip() for l in t.split('\n') if l.strip()]
    tail=lines[-1] if lines else ''
    # token-free period search
    best=None
    for p in range(8,400):
        if len(t)>=3*p and t[-p:]==t[-2*p:-p]==t[-3*p:-2*p]:
            best=p;break
    if best is None: return None,None
    # how far back does the repetition go
    k=3
    while len(t)>=(k+1)*best and t[-(k+1)*best:-k*best]==t[-best:]: k+=1
    return best,len(t)-k*best
cut=[x for x in r if x['cut']]
nf=[x for x in r if not x['cut']]
info=[loop_info(x['text']) for x in cut]
lp=[(x,i) for x,i in zip(cut,info) if i[0] is not None]
print('cut',len(cut),'finished',len(nf),'acc finished',round(np.mean([x['correct'] for x in nf]),3),'correct among cut',sum(x['correct'] for x in cut))
print('style among cut',{k:sum(x['text'].startswith(k) for x in cut) for k in ['Let','Here']})
print('exact periodic tail',len(lp),'period med',np.median([i[0] for _,i in lp]) if lp else None)
print('cut with gold anywhere',sum(has_gold(x['text'],x['gold']) for x in cut),
      'gold before loop start',sum(has_gold(x['text'][:i[1]],x['gold']) for x,i in lp),
      'finished wrong w gold anywhere',sum(has_gold(x['text'],x['gold']) for x in nf if not x['correct']),'of',sum(not x['correct'] for x in nf))
print('fraction of text before loop med',round(np.median([i[1]/len(x['text']) for x,i in lp]),2) if lp else None)
for x,i in lp[:2]:
    print('--',x['id'],i,repr(x['text'][max(0,i[1]-90):i[1]+150])[:300])
nl=[x for x,i in zip(cut,info) if i[0] is None]
print('non-periodic cut',len(nl))
for x in nl[:2]:
    print('--',x['id'],repr(x['text'][-220:])[:300])
print('used_marker rate',round(np.mean([x['used_marker'] for x in nf]),3),'acc marker',round(np.mean([x['correct'] for x in nf if x['used_marker']]),3),'acc nomarker',round(np.mean([x['correct'] for x in nf if not x['used_marker']]),3))
```