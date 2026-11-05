I'll run the paired comparison, and this time print the per-token equivalents for E1 and E2 next to R and C, using each loop line's actual length so the selectivity comparison is fair. I'll do the same for pass 21. The question is whether E falls clearly more per token than R, digits and C.

```python
r22=screen(22,'L5h2 L11h1 L17h0,2,3 x0.5')
LL=[]
for r in idx2:
    if r['group'] not in ('R','C','W'):
        for q,(a,b) in enumerate(r['spans'][:3]): LL.append((q,b-a+1))
LL=np.array(LL); ml=[LL[LL[:,0]==q,1].mean() for q in range(3)]
print('mean line tokens',np.round(ml,1))
for lab,r in [('p21',r21),('p22',r22)]:
    print(lab,'per token E0,E1,E2',[round(r[f'E{q}'][0]/ml[q],4) for q in range(3)],'R',r['R_all'][0],'dig',r['R_dig'][0],'C',r['C'][0],'E_all',r['E_all'][0])
```