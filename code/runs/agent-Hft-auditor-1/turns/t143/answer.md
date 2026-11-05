The commit-earlier idea is dead, and the curve is clear. Forcing a stop never beats the baseline by more than +0.20 ± 0.6, and stopping before about 600 tokens loses accuracy, because the answers still running at that point include real solutions that haven't finished. Among answers that end up cut, the current last number equals the gold answer only 0 to 7% of the time at every checkpoint. So these answers aren't circling near a right answer; they're lost throughout. That closes commit-earlier, and the position-aware unit idea with it.

If those problems can be rescued at all, it has to come from a different route, not a better ending. That gives a question I can answer for free. There are 85 Edit A cut-offs on dev_b, and I have answers to those same problems from arms that opened differently: original (`Here`), `First`, `To`, and `Okay`, whose list format has only about 2% cut-offs. If those paths solve a good share of them, the `Let` path is failing specifically there, and the problem isn't simply too hard. That would open a real option: when `Let` starts to fail, a restart in another style might succeed. It would also show whether the paths' strengths complement each other enough for problem-specific routing to pay off. If those arms also fail on these 85 problems, then they're just hard for the model and the cut class is out of reach.

I'll also compute how far Edit A sits from what's possible on dev_b: the share of problems solved by at least one arm, and by a majority of arms.

```python
import numpy as np, json, collections
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
DB=json.load(open('/work/dev_b.json'))
names={5:'orig',6:'EditA',7:'a1.5',8:'a.5',10:'B+',11:'B-',13:'First',14:'p14',15:'cw',16:'chan0',17:'chan-1',19:'Okay',20:'To',23:'vbos',26:'n180x2'}
A={k:rd('/work/gen_%d.jsonl'%k) for k in names}
for k in (13,14,19,20,5):
    o=collections.Counter(A[k][i]['text'].lstrip().split()[0].strip(',:')[:8] if A[k][i]['text'].strip() else '' for i in DB)
    print(names[k],'acc %.2f cut %.2f tokens %.0f open %s'%(100*np.mean([A[k][i]['correct'] for i in DB]),100*np.mean([A[k][i]['cut'] for i in DB]),np.mean([A[k][i]['n_tokens'] for i in DB]),[(x.encode('ascii','replace').decode(),n) for x,n in o.most_common(3)]))
g=A[6]
cutids=[i for i in DB if g[i]['cut']]; wrongfin=[i for i in DB if not g[i]['cut'] and not g[i]['correct']]; right=[i for i in DB if g[i]['correct']]
print('EditA groups: right',len(right),'finished wrong',len(wrongfin),'cut',len(cutids))
print('arm | acc on EditA-cut | cut rate there | acc on EditA-finished-wrong | acc on EditA-right')
for k in (5,13,14,19,20):
    f=lambda S,key:100*np.mean([A[k][i][key] for i in S])
    print('%-6s | %5.1f | %5.1f | %5.1f | %5.1f'%(names[k],f(cutids,'correct'),f(cutids,'cut'),f(wrongfin,'correct'),f(right,'correct')))
LetArms=[6,7,8,10,11,15,16,17,23,26]
print('other Let arms on EditA-cut: acc',round(100*np.mean([A[k][i]['correct'] for k in LetArms[1:] for i in cutids]),1),'cut',round(100*np.mean([A[k][i]['cut'] for k in LetArms[1:] for i in cutids]),1))
M=np.array([[A[k][i]['correct'] for k in names] for i in DB],float)
L=np.array([[A[k][i]['correct'] for k in LetArms] for i in DB],float)
print('all %d arms: any right %.1f, majority right %.1f, all right %.1f, none right %.1f'%(M.shape[1],100*(M.sum(1)>0).mean(),100*(M.mean(1)>.5).mean(),100*(M.mean(1)==1).mean(),100*(M.sum(1)==0).mean()))
print('Let arms only (%d): any %.1f, majority %.1f, all %.1f, none %.1f, mixed %.1f'%(L.shape[1],100*(L.sum(1)>0).mean(),100*(L.mean(1)>.5).mean(),100*(L.mean(1)==1).mean(),100*(L.sum(1)==0).mean(),100*((L.mean(1)>0)&(L.mean(1)<1)).mean()))
p=L.mean(1); 
print('mean per-arm acc in Let arms %.1f | histogram of per-problem right rate (0,.1..,1):'%(100*p.mean()),np.histogram(p,bins=[-.01,.05,.25,.5,.75,.95,1.01])[0])
ok=A[19]; let=A[6]
b=np.array([let[i]['correct'] for i in DB],float); c=np.array([ok[i]['correct'] for i in DB],float)
print('Let vs Okay: both %.1f, Let only %.1f, Okay only %.1f, neither %.1f | oracle choice %.1f'%(100*(b*c).mean(),100*(b*(1-c)).mean(),100*((1-b)*c).mean(),100*((1-b)*(1-c)).mean(),100*np.maximum(b,c).mean()))
```