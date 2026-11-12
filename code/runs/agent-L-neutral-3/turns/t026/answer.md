Both results are informative. Stopping loops at their onset is not worthwhile: greedy reproduces the gold number in only 6 of 184 probes (3.3%), so loops are mostly a symptom of problems the model has already lost, with a payoff of under +1 point. I'm closing that line, even though it's the most visible failure. In reference-solution prefixes, arithmetic turns out to be fairly healthy, with result digits correct 92-94% of the time, and only 71 of 800 solutions break first at a result. The largest first break by far is the choice of which number from the problem to use next: 357 of 800 solutions break there, with the first digit correct only 78% of the time. So the main failure with a correct prefix is retrieving the right quantity, which fits the dropped entities and swapped quantities I saw when reading failures. Some of this is ambiguity in operand order, so I need to see how structured it is. I'll break retrieval accuracy down by whether the target number has already been used in the solution, and by its position in the problem. I'll also check what the model's wrong top digit points to. A strong bias, such as avoiding unused numbers or preferring recent ones, would suggest a specific attention mechanism to examine next. If errors look uniform, I'd treat them as general retrieval noise.

```python
open('/work/notes.md','a').write("""
Pass 4: loop-stop probes greedy gold only 6/184 (3.3%): LOOP-STOP LINE CLOSED. Reference prefixes: result digits top correct .92-.94; first break: fromproblem 357, othernum 173, fromearlier 96, result 71, none 103 of 800. Main mechanism candidate: retrieving next operand from the problem.
""")
tab=collections.defaultdict(lambda:[0,0])
wrongkind=collections.Counter()
for i,t in enumerate(tm):
    if t['kind']!='ref': continue
    ids_=[int(x) for x in ii4[i,:int(L4[i])]]; p=t['plen']
    prob=byid[t['id']]['problem']
    pn=[x.replace(',','').rstrip('.') for x in re.findall(r'\d[\d,]*\.?\d*',prob)]
    uniq=list(dict.fromkeys(pn))
    used=set(); q=p;K=len(ids_)
    last_used_idx=None
    while q<K:
        if ids_[q] in DIG:
            r=q
            while r<K and ids_[r] in DIG: r+=1
            s=''.join(DIG[ids_[x]] for x in range(q,r))
            before=Dc(ids_[max(p,q-3):q])
            if '=' not in before[-3:] and s in uniq:
                pos=uniq.index(s); fresh=s not in used; n_=len(uniq)
                tops=[int(a) for a in ti4[i,q-1]]; dg=[a for a in tops if a in DIG]
                good=bool(dg) and dg[0]==ids_[q]
                frac='early' if pos<n_/3 else ('mid' if pos<2*n_/3 else 'late')
                tab[('fresh' if fresh else 'reused',frac)][0]+=1; tab[('fresh' if fresh else 'reused',frac)][1]+=good
                tab[('nuniq',min(n_,6))][0]+=1; tab[('nuniq',min(n_,6))][1]+=good
                if not good and dg:
                    dtop=DIG[dg[0]]
                    cand=[u for u in uniq if u.startswith(dtop)]
                    usedc=[u for u in cand if u in used]; freshc=[u for u in cand if u not in used]
                    k_='no_problem_number' if not cand else ('only_used' if usedc and not freshc else ('only_fresh' if freshc and not usedc else 'both'))
                    wrongkind[k_]+=1
                used.add(s)
            elif s in uniq: used.add(s)
            q=r
        else: q+=1
for k,v in sorted(tab.items(),key=str): print(k,'n',v[0],'acc',round(v[1]/v[0],3))
print(wrongkind.most_common())
```