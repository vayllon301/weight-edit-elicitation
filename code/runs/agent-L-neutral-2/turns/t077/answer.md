The cut reduction replicated exactly, from 23.8% to 10.2%, but the accuracy gain didn't: **-2.34 ± 2.65 on these problems**, with 20 gained and 26 lost. Pooled across p7 and p8 (512 answers), L2 comes to -0.39 ± 1.77. That's a solid causal result on loops with no demonstrated accuracy benefit. This is what the task warned about, and I won't report the cut rate as a success. L2 as built doesn't beat the original.

Before rejecting the whole idea, I want to know *why*, because a specific fixable side effect could be hiding the gain. The obvious suspect is the Let-to-Here flips. On p8, Let answers were doing well (27% in the earlier pass), and the edit pushes many of them into the weaker style. If the loss sits in that flipped group, while non-flipped groups are flat or positive, then an answer-only write that leaves the opener alone is a principled fix, not a post-hoc tweak. That's because loops only form during answers, so there's no reason to touch the prompt position at all. If losses are spread evenly, I'll treat this as evidence that loop removal doesn't help and close it.

```python
breakdown(12)
open('/work/notes.md','a').write("\npass11 L2 alpha-1 p7: cut .262->.102, acc +1.56+-2.35 (escapes 50, WR2 RW1). pass12 L2 p8: cut .238->.102, acc -2.34+-2.65 (WR20 RW26). Pooled L2 n=512: -.39+-1.77. Loop removal causal, accuracy not improved so far.\n")
o,e=loadg(12); o2,e2=loadg(11)
S=[]
for O_,E_ in [(o2,e2),(o,e)]:
    for i in O_:
        S.append(dict(id=i,so=st(O_[i]['text']),se=st(E_[i]['text']),co=O_[i]['correct'],ce=E_[i]['correct'],cuto=O_[i]['cut'],cute=E_[i]['cut']))
S=pd.DataFrame(S)
G=S.groupby(['so','se']).agg(n=('co','size'),acc0=('co','mean'),acc1=('ce','mean'),cut0=('cuto','mean'),cut1=('cute','mean'),WR=('ce',lambda s:0),).round(3)
G['delta_pts']=((G.acc1-G.acc0)*100).round(1); print(G.drop(columns='WR').to_string())
S['d']=S.ce.astype(int)-S.co.astype(int)
for (a,b),q in S.groupby(['so','se']):
    print(a,b,'paired %.1f +- %.1f'%(100*q.d.mean(),100*q.d.std()/np.sqrt(len(q))))
q=S[S.so==S.se]; print('non-flipped overall paired %.1f +- %.1f n %d'%(100*q.d.mean(),100*q.d.std()/np.sqrt(len(q)),len(q)))
print(S.groupby(['cuto','cute']).agg(n=('d','size'),d=('d','sum'),WR=('d',lambda s:(s>0).sum()),RW=('d',lambda s:(s<0).sum())))
```