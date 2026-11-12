To keep using the remaining passes on something specific, I'll first test a hypothesis that needs no pass. My readings described "dropped conditions": wrong setups that ignore part of the question. If that is a real failure type, wrong finished answers should leave out more of the question's numbers than correct finished answers do, even after controlling for difficulty. To control for that, I'll compare within problems the model got right in one pass and wrong in another, and also across all DEV answers matched by the number count of the question. A clear gap, say wrong answers missing 15+ points more of the question's numbers, would make "unused question quantities" a target. That target would be tracking which numbers are already used, a different mechanism from the fetch heads, which only read numbers. No gap closes the last setup hypothesis.

```python
def qnums(pr): return set(x.replace(',','') for x in re.findall(r'\d[\d,]*\.?\d*', pr) if x.replace(',','').replace('.','').isdigit())
def used_frac(r):
    qn=qnums(prac[r['id']]['problem']); 
    if not qn: return None
    tn=set(x.replace(',','') for x in re.findall(r'\d[\d,]*\.?\d*', r['text']))
    return np.mean([q in tn for q in qn]), len(qn)
fin=[r for r in rows if not r['cut']]
res={True:[],False:[]}
for r in fin:
    u=used_frac(r)
    if u: res[bool(r['correct'])].append(u)
out={}
for c in (True,False):
    a=np.array(res[c]); out[c]=(len(a), round(a[:,0].mean(),3), round(np.mean(a[:,0]<1),3))
# matched by number count
bins={}
for c in (True,False):
    for f,n in res[c]: bins.setdefault(min(n,6),{True:[],False:[]})[c].append(f)
mb={k:(len(v[True]),round(np.mean(v[True]),3) if v[True] else None,len(v[False]),round(np.mean(v[False]),3) if v[False] else None) for k,v in sorted(bins.items())}
# discordant within-problem (pair_texts: correct, wrong alternating)
d=[]
for j in range(0,len(pt),2):
    qn=qnums(prac[pm[j]['id']]['problem'])
    if not qn: continue
    f=lambda t: np.mean([q in set(x.replace(',','') for x in re.findall(r'\d[\d,]*\.?\d*', resp(t))) for q in qn])
    d.append(f(pt[j])-f(pt[j+1]))
(out, mb, (len(d), round(np.mean(d),3), round(np.std(d)/np.sqrt(len(d)),3)))
```