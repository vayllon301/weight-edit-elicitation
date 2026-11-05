Three results here change my plan.

**1. Generation is exactly reproducible on `dev_a`.** gen_1 and gen_2 are the same text on all 512 problems. So a 512-answer "edited" pass on `dev_a`, in the same order, is fully paired with the stored baseline, with about twice the answers of a "both" pass. I'll still use "both" for formal HOLDOUT confirmation, as the task specifies.

**2. E1's digit gain looks like more than temperature.** Using p > 0.5 as a proxy for the top token, the share of correct result tokens goes 0.863 → 0.905 (E1) and 0.871 (V1), against 0.919 in raw. E1 rescues 36% of the failing tokens and loses under 1% of the good ones. So the real part of the effect sits outside volume, and arithmetic does get better. That makes E1's lack of accuracy gain more puzzling.

**3. The chat shift at layer 12 is close to one constant vector.** About 70% of its energy is the mean shift, and the largest remaining component is only 12%. That fits a constant broadcast, and a single direction is something a precise edit can remove.

Earlier numbers also suggest E1 does two opposing things. On originally cut problems it rescues about 15%, roughly +3.9 points overall. On the other 376 problems it scores about 24% against about 33%, roughly -6.4 points.

Before designing the split, I want a careful count from my stricter arithmetic checker to confirm the arithmetic improvement, plus a look at how the original's loops actually begin. If loops start after a final answer has already been given, stopping is the issue. If they start in mid-reasoning, that's a different mechanism.

```python
import collections
def arith_table(gs, sel, name):
    allr=[]
    for i in sel:
        seen=set()
        for e in check2(gs[i]['text']):
            if e['expr'] in seen: continue
            seen.add(e['expr']); e['id']=i; allr.append(e)
    A=pd.DataFrame(allr)
    by=A.groupby('kind').ok.agg(['size','mean']).round(3)
    per=A.groupby('id').ok.apply(lambda s:(~s).any())
    print(name,'exprs',len(A),'err %.4f answers-with-exprs %d, with>=1 error %.3f'%(1-A.ok.mean(),len(per),per.mean()),by.to_dict('index'))
    return A
fin1=[i for k,i in enumerate(ids) if not cut1[k]]; fin7=[i for k,i in enumerate(ids) if not cut7[k]]
both_fin=[i for k,i in enumerate(ids) if (not cut1[k]) and (not cut7[k])]
A1=arith_table(g1,ids,'orig all'); A7=arith_table(g7,ids,'E1 all')
_=arith_table(g1,fin1,'orig finished'); _=arith_table(g7,fin7,'E1 finished')
_=arith_table(g1,both_fin,'orig both_fin'); _=arith_table(g7,both_fin,'E1 both_fin')
_=arith_table(g1,[i for k,i in enumerate(ids) if cut1[k]],'orig cut'); _=arith_table(g7,[i for k,i in enumerate(ids) if cut1[k]],'E1 on orig-cut')
# how many wrong answers contain an error, and do error-free answers score better?
for nm,gs,A in [('orig',g1,A1),('E1',g7,A7)]:
    bad=set(A[~A.ok].id); hasx=set(A.id)
    a=[gs[i]['correct'] for i in ids if i in hasx and i not in bad and not gs[i]['cut']]; b=[gs[i]['correct'] for i in ids if i in bad and not gs[i]['cut']]
    print(nm,'finished & error-free acc %.3f (n=%d); finished & >=1 err acc %.3f (n=%d)'%(np.mean(a),len(a),np.mean(b),len(b)))

# where loops start in original cut answers: first char position of repeated line, and whether a final answer came before
def loop_info(t):
    lines=[l.strip() for l in t.split('\n') if len(l.strip())>12]
    seen={}; start=None
    for n_l,l_ in enumerate(lines):
        if l_ in seen: start=seen[l_]; break
        seen[l_]=n_l
    pre='\n'.join(lines[:start]) if start is not None else ''
    return start, len(lines), bool(re.search(r'####|boxed|final answer',pre,re.I)), (len(pre)/max(1,len(t)))
cuts=[i for k,i in enumerate(ids) if cut1[k]]
inf=[loop_info(g1[i]['text']) for i in cuts]
print('orig cut n',len(cuts),'with repeated line %.3f, final-answer-before-loop %.3f, median frac before loop %.3f'%(np.mean([x[0] is not None for x in inf]),np.mean([x[2] for x in inf]),np.median([x[3] for x in inf if x[0] is not None])))
for i in cuts[:4]:
    t_=g1[i]['text']; print('--',i,'gold',g1[i]['gold'],'|',t_[len(t_)//3:len(t_)//3+150].replace('\n',' / ').encode('ascii','replace').decode(),'...END:',t_[-90:].replace('\n',' / ').encode('ascii','replace').decode())
```