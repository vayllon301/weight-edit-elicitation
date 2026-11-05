D2 scored 58.0% on `dev_a` against T4's 58.9%, so its edge didn't carry over: about -0.6 paired, on a set that leans against it. Together with +2.15 on `dev_c512`, the pooled gain looks close to +0.8, and I expect that to fall short of the one-standard-error rule, which would keep T4. Cut-offs did drop again (11.9% to 10.6%), so D2's lower looping seems to repeat even though accuracy doesn't follow. That fits what I've seen throughout: fewer cut-offs doesn't mean more correct answers.

Next I'll compute the exact paired and pooled numbers and apply the rule as written. I'll also split D2 vs T4 into finished answers and cut changes on both sets, to see whether the same thing happened on each. Then I'll read the rest of my notes, which I need before planning the five free passes. They should show whether L11 h1 and the L10 dose were tested, and which lines are closed.

If the pooled gain is below one SE, T4 goes back as the candidate and D2 is reported as a tie with a modest effect on loops. The a=1.2 point then won't count as an improvement, but the dose curve stays as causal evidence for the mechanism.

```python
ga, gb = rd('/work/gen_23.jsonl'), rd('/work/gen_41.jsonl')
g32, g39 = rd('/work/gen_32.jsonl'), rd('/work/gen_39.jsonl')
ia, ic = list(ga), list(g32)
def dv(a,b,ids): return np.array([float(b[i]['correct'])-float(a[i]['correct']) for i in ids])
print('D2 vs T4 dev_a', pair(ga,gb,ia), 'fifths', [round(100*dv(ga,gb,ia[k*len(ia)//5:(k+1)*len(ia)//5]).mean(),1) for k in range(5)])
print('D2 vs T4 dev_c512', pair(g32,g39,ic))
d = np.concatenate([dv(ga,gb,ia), dv(g32,g39,ic)])
print('POOLED D2-T4 n %d mean %.2f se %.2f wins %d losses %d'%(len(d),100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum()))
for nm,A,B,ids in (('dev_a',ga,gb,ia),('dev_c512',g32,g39,ic)):
    f = [i for i in ids if not A[i]['cut'] and not B[i]['cut']]
    gone = [i for i in ids if A[i]['cut'] and not B[i]['cut']]
    new = [i for i in ids if B[i]['cut'] and not A[i]['cut']]
    c = lambda G: (sum(G[i]['text'].lstrip().startswith('Let') for i in ids), sum(G[i]['text'].lstrip().startswith('Here') for i in ids))
    print(nm,'both finished',len(f),pair(A,B,f),'| cuts gone',len(gone),'right now',sum(B[i]['correct'] for i in gone),'| new',len(new),'right in T4',sum(A[i]['correct'] for i in new),'| Let/Here T4',c(A),'D2',c(B),'| cut %.3f->%.3f'%(np.mean([A[i]['cut'] for i in ids]),np.mean([B[i]['cut'] for i in ids])))
txt = open('/work/notes.md').read()
s = txt.find('pass 24 result T5')
e = txt.find('pass 30 result T11')
print(len(txt), s, e)
print(txt[s:e][:7500])
```