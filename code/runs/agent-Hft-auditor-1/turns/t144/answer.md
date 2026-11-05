The complementarity table gives a clear answer. On the 85 cut problems, the other paths solve only 22 to 25%, and `Let` arms themselves solve 14.6%, so these are mostly hard problems for this model on any path. Even a perfect rescue would add about 3.7 points. A restart strategy couldn't reach that ceiling, because it would need to work after a failed attempt and only on these problems. Across the 10 `Let` arms, 42% of problems are always right, 20% are never right, and 38% are mixed. The majority-vote score (60.2) essentially equals the mean arm (61.1). So for this model, mixed problems are about as likely to come out right as wrong, with no hidden systematic pull toward the right answer that a stronger signal could expose.

I also noticed something that needs checking. The arm I labelled `First` (gen_13) at its weaker dose opened `First` on only 116 problems and `Let` on 373, and it scored 62.70, the highest number on dev_b so far. The fully forced `First` arm scored only 60.16. That pattern could be noise, but it could also mean a partial write selects the problems where `First` does better. That would be the problem-specific routing I've closed for `Let`/`Here`, now appearing with a different alternative. It's a free check. I'll compare `Let` and `First` answers on those same 116 problems, and split all forced-opening arms by n180 tertile, since that's a validated natural signal of problem ease. If some path wins on hard problems and another on easy ones, routing has real value. If `Let` leads or ties in every tertile and the 116-problem difference is within noise, then I'll close routing as well.

```python
import numpy as np, json, collections
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
ids=[json.loads(l)['id'] for l in open('/work/gen_5.jsonl')]
names={5:'orig',6:'EditA',13:'gen13',14:'FirstF',19:'Okay',20:'To'}
A={k:rd('/work/gen_%d.jsonl'%k) for k in names}
C={k:np.array([A[k][i]['correct'] for i in ids],float) for k in names}
CT={k:np.array([A[k][i]['cut'] for i in ids],bool) for k in names}
op13=np.array([A[13][i]['text'].lstrip().split()[0].strip(',:') for i in ids])
op14=np.array([A[14][i]['text'].lstrip().split()[0].strip(',:') for i in ids])
F=op13=='First'; print('gen13 First problems',int(F.sum()))
def pd(a,b,k):
    d=a[k]-b[k]; return 100*d.mean(),100*d.std(ddof=1)/np.sqrt(k.sum())
print('on gen13-First problems: gen13 %.1f EditA %.1f FirstF %.1f orig %.1f | gen13-EditA %+.1f se %.1f | FirstF-EditA %+.1f se %.1f'%(100*C[13][F].mean(),100*C[6][F].mean(),100*C[14][F].mean(),100*C[5][F].mean(),*pd(C[13],C[6],F),*pd(C[14],C[6],F)))
print('on remaining: gen13 %.1f EditA %.1f | diff %+.1f se %.1f | FirstF-EditA %+.1f se %.1f'%(100*C[13][~F].mean(),100*C[6][~F].mean(),*pd(C[13],C[6],~F),*pd(C[14],C[6],~F)))
print('all: gen13-EditA %+.2f se %.2f'%pd(C[13],C[6],np.ones(len(ids),bool)))
same=F&(op14=='First')
print('First-path within: gen13 vs FirstF text identical share on F %.2f'%np.mean([A[13][i]['text']==A[14][i]['text'] for i,k in zip(ids,F) if k]))
mp=np.load('/work/gen_5_acts/mlp_last_prompt.npy',mmap_mode='r'); lp=np.array(mp[:,16,180]).astype(np.float32)
print('n180 mean: gen13 First %.2f others %.2f'%(lp[F].mean(),lp[~F].mean()))
q=np.quantile(lp,[1/3,2/3]); T=[lp<q[0],(lp>=q[0])&(lp<q[1]),lp>=q[1]]
print('tertile: EditA FirstF To Okay orig (acc) | cut same order | n')
for k,n in zip(T,('most neg','mid','least neg')):
    print('%-9s'%n,' '.join('%5.1f'%(100*C[a][k].mean()) for a in (6,14,20,19,5)),'|',' '.join('%5.1f'%(100*CT[a][k].mean()) for a in (6,14,20,19,5)),'|',int(k.sum()))
for a,nm in ((14,'FirstF'),(20,'To'),(19,'Okay')):
    print(nm,'minus EditA by tertile:',' '.join('%+.1f(%.1f)'%pd(C[a],C[6],k) for k in T))
```