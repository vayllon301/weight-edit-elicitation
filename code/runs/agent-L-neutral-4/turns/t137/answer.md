The block failed at the pair step only: pair_meta has one entry per pair (77), not one per text, so the index has to be j//2. The first two parts of the analysis are unaffected. I'll rerun with that fix. The hypothesis and the decision rule stay the same.

```python
d=[]
for j in range(0,len(pt),2):
    qn=qnums(prac[pm[j//2]['id']]['problem'])
    if not qn: continue
    f=lambda t: np.mean([q in set(x.replace(',','') for x in re.findall(r'\d[\d,]*\.?\d*', resp(t))) for q in qn])
    d.append(f(pt[j])-f(pt[j+1]))
(len(pm), out, mb, (len(d), round(np.mean(d),3), round(np.std(d)/np.sqrt(len(d)),3)))
```