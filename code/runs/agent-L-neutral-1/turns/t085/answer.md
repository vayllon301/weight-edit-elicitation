The spurious-constant idea fails. Only 5.5% of finished answers contain a spurious operand, and overall the model brings in far fewer unstated numbers than the references do, so the examples I read weren't representative.

The same table shows something I hadn't weighed: **the cut rate doesn't fall on easy problems**. It's 34% at 1 step, 23% at 2 and 22–33% at 5–7. If loops only came from getting lost in hard problems, short problems should almost never loop. My "loops are a symptom" verdict also rested on forced exits at the loop's start, and that can't show what the model would have done had it never entered the loop. So that closure was too strong, and I'm reopening the question.

Next I'll locate each loop and measure how early it starts, on 1–2 step problems especially. I'll also check what the text is doing as it enters the loop and read a few entries. If loops start early, before the reasoning has failed, then preventing them could recover much more than the exit test suggested. If they start only after the model has already gone wrong, the closure stands.

```python
import numpy as np, collections, re
def loop_info(ids, minrep=3, maxp=220):
    n=len(ids)
    for p in range(2,maxp):
        if n<p*minrep: break
        if ids[n-p:]==ids[n-2*p:n-p]==ids[n-3*p:n-2*p] or ids[n-p:]==ids[n-2*p:n-p] and ids[n-2*p:n-p]==ids[n-3*p:n-2*p]:
            s=n-1
            while s-p>=0 and ids[s]==ids[s-p]: s-=1
            return s-p+1,p
    return None
rows=[]; ex={}
for k,d in orig.items():
    if not d['cut']: continue
    ids=enc(d['text']); L=loop_info(ids[:-1] if len(ids)>2 else ids)
    if L is None:
        # trailing partial cycle: trim up to maxp tokens
        for tr in range(1,200,7):
            L=loop_info(ids[:-tr])
            if L: break
    st=min(steps(P[k]),6)
    if L is None:
        rows.append(dict(id=k,steps=st,loop=False,start=np.nan,period=np.nan,eq_before=np.nan,n=len(ids))); continue
    s,p=L
    pre=tok.decode(ids[:s]); cyc=tok.decode(ids[s:s+p])
    rows.append(dict(id=k,steps=st,loop=True,start=s,period=p,eq_before=len(re.findall(r'=',pre)),n=len(ids),gold_in_pre=bool(re.search(r'(?<![\d.])'+re.escape(str(d['gold']).replace('.0',''))+r'(?![\d])',pre.replace(',','')))))
    ex[k]=(pre,cyc)
L=pd.DataFrame(rows); L.to_pickle('/work/loop_info.pkl')
print('cut answers',len(L),'with loop found',int(L.loop.sum()))
print(L[L.loop].groupby('steps')[['start','period','eq_before']].median().round(1).assign(n=L[L.loop].groupby('steps').size(), gold_pre=L[L.loop].groupby('steps').gold_in_pre.mean().round(3)))
print('start quantiles',L.start.quantile([.1,.25,.5,.75,.9]).round(0).to_dict())
print('period quantiles',L.period.quantile([.1,.25,.5,.75,.9]).round(0).to_dict())
first=collections.Counter()
for k,(pre,cyc) in ex.items():
    w=re.findall(r"[A-Za-z']+|\d|\$|\\[a-z]+|\*\*|\S",cyc.strip())
    first[(w[0] if w else '?')]+=1
print('cycle first symbol',first.most_common(12))
easy=[k for k in ex if min(steps(P[k]),6)<=2]
random.seed(3); random.shuffle(easy)
for k in easy[:7]:
    pre,cyc=ex[k]; r=L[L.id==k].iloc[0]
    print('---',k,'steps',r.steps,'start',int(r.start),'period',int(r.period),'gold',orig[k]['gold'])
    print('Q:',P[k]['problem'][:140].replace('\n',' '))
    print('PRE:',pre[-170:].replace('\n',' / '))
    print('CYC:',cyc[:130].replace('\n',' / '))
```