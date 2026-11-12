This is the first non-null asymmetry. On 293 identical equations, the model is right 94.9% of the time in a bare "Calculate" prompt but only 87.4% inside the gold word-problem solution: −7.5 points, −0.22 ± 0.06 nats. 26 equations fail only in context, against 4 that fail only bare.

There is a harmless explanation to rule out first. Inside a GSM8K solution, the token right after "=" can legitimately be something other than a digit ("$", a space, LaTeX). My exact-match criterion counts any such top-1 as a failure even if the model's arithmetic is fine. I'll examine the 37 in-context failures, classifying the model's top-1 token at the first wrong result position as a different digit or a non-digit (dollar sign, whitespace, other), without printing raw vocabulary. For digit errors I'll show the equation, the gold digit and the predicted digit.
- Mostly non-digit predictions would make the gap a formatting artefact.
- Mostly wrong digits would make it a genuine context-induced arithmetic failure, worth localizing, for example as interference from other numbers in the problem.

```python
def ctx_pos(k):
    pid,a,op,b,c=probes[k]; j=jpos[pid]; sol=byid[pid]['solution']
    m=next(mm for mm in eqre2.finditer(sol) if mm.groups()==(a,op,b,c))
    cpre=ann.sub('',sol[:m.end()]); cs=clean_sol(sol); base=TS[j].find(cs[:60]); st=base+len(cpre); en=st+len(c); enc=tok.encode(TS[j])
    return j,np.array([kk-1 for kk,(s,e) in enumerate(enc.offsets) if kk>=1 and e>s and s<en and e>st])
cat_=collections.Counter(); ex_=[]
for (k,lpc,exc,lpb,exb,op) in rows_:
    if exc: continue
    j,pos=ctx_pos(k); bad=[p for p in pos if tk3[j,p-1,0]!=ids3[j,p]][0]
    s_=tok.decode([int(tk3[j,bad-1,0])]); g_=tok.decode([int(ids3[j,bad])])
    kind='digit' if s_.strip().isdigit() else ('dollar' if '$' in s_ else ('space' if s_.strip()=='' else 'other'))
    cat_[(kind,'bare_ok' if exb else 'bare_wrong')]+=1
    if kind=='digit' and len(ex_)<8: ex_.append(f"{probes[k][1]}{probes[k][2]}{probes[k][3]}={probes[k][4]} gold_digit {g_} pred {s_.strip()} (idx {list(pos).index(bad)})")
print(dict(cat_)); print('\n'.join(ex_))
```