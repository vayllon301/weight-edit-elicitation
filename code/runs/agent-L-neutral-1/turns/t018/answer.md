The arithmetic picture is now clearer and less alarming. Clean two-operand results are about 94–97% correct, while multi-operation expressions manage only 45%, and those are often written without the intermediate step. Large products and quotients are the weak spot (around 82% for 4+ digits). That's a real but modest loss: about 5% per equation over about 3 equations would cost perhaps 15% of otherwise-correct answers. So arithmetic is a contributor, not the main cause of the 19% score.

Before committing to a mechanism I need a better breakdown of the failures. Three checks with existing data:
- whether accuracy collapses with step count, which would suggest working-memory or planning failure rather than uniform weakness
- how often problem numbers are dropped from the answer, or a wrong number appears, which would suggest a retrieval failure
- the 24 wrong answers where the gold value appears in the last 200 characters, because those may be right answers lost to formatting, which would be a cheap and specific fix

```python
A_rows=[]
for x in g:
    pr=pid[x['id']]
    nsteps=len(re.findall(r'<<',pr['solution']))
    pn=[m.replace(',','').rstrip('.') for m in re.findall(r'\d[\d,]*\.?\d*',pr['problem'])]
    tn=[m.replace(',','').rstrip('.') for m in re.findall(r'\d[\d,]*\.?\d*',x['text'])]
    miss=np.mean([n not in tn for n in pn]) if pn else 0
    A_rows.append(dict(id=x['id'],steps=nsteps,plen=len(pr['problem']),miss=miss,correct=x['correct'],cut=x['cut']))
P=pd.DataFrame(A_rows); P['stepb']=P.steps.clip(upper=6)
print(P.groupby('stepb').agg(n=('correct','size'),acc=('correct','mean'),cut=('cut','mean'),miss=('miss','mean')).round(3).T.to_string())
print('missing-number share: correct %.3f wrong %.3f'%(P[P.correct].miss.mean(),P[~P.correct].miss.mean()))
print('corr steps~acc',round(np.corrcoef(P.steps,P.correct)[0,1],3))
# wrong answers whose gold appears late
late=[]
for x in g:
    if x['correct'] or x['cut']: continue
    tail=x['text'][-220:]
    ns=[]
    for m in re.findall(r'\d[\d,]*\.?\d*',tail):
        try: ns.append(f(m))
        except: pass
    try: gv=float(str(x['gold']).replace(',',''))
    except: continue
    if any(abs(n-gv)<1e-6 for n in ns): late.append(x)
print('wrong finished with gold in tail',len(late))
for x in late[:12]:
    print(x['id'],'gold',x['gold'],'pred',x['pred'],'mk',x['used_marker'],'|',x['text'][-110:].replace('\n',' / ').encode('ascii','replace').decode())
```