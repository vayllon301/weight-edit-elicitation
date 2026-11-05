The persistent-write tool and the activation files are all available. This gives me what I need to build the path direction without a pass. I'll compute it from dev_a only, using pass 1's token-averaged `h18`: finished natural `Let` answers minus finished natural `Here` answers, stratified by problem step count. That keeps problem difficulty from leaking in. Excluding cut answers also keeps long doubt passages from defining the direction.

Three checks come before any pass, all on dev_b data that didn't build the direction:

- **Separation:** does the direction separate `Let` from `Here` answers on dev_b? If not, it isn't a real path axis.
- **Novelty:** what's the overlap with the opening direction `d18` and with `d_cw`? A very high overlap with `d18` would mean I'm only repeating Edit A.
- **Dose within `Let`:** among `Let` answers, does a higher projection go with being more correct inside step strata? This is observational, so it can't prove a lever, but a strong negative trend would warn against pushing this way.

I'll also print the size of the path gap relative to the spread within `Let`, so that I can set the write strength by hand from the path gap itself rather than by trial. Nothing is saved to the model in this step.

```python
import numpy as np, json, re
need = ['summ','units17','m17','w17_','rbar17','NAME17','H1','H5','m1','m5','ok1','ok5','st1','st5','tk5','strat_auc']
print({k:(k in globals()) for k in need})
g1 = [json.loads(l) for l in open('/work/gen_1.jsonl')]
g5 = [json.loads(l) for l in open('/work/gen_5.jsonl')]
H1 = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r'); H5 = np.load('/work/gen_5_acts/hidden_mean_answer.npy', mmap_mode='r')
print('H1', H1.shape, 'H5', H5.shape, 'rows', len(g1), len(g5))
print('ok1 aligned', bool(np.array_equal(np.asarray(ok1,bool), np.array([r['correct'] for r in g1],bool))), 'ok5 aligned', bool(np.array_equal(np.asarray(ok5,bool), np.array([r['correct'] for r in g5],bool))))
def op(r):
    m = re.match(r"\s*([A-Za-z']+)", r['text']); return m.group(1) if m else '?'
o1 = np.array([op(r) for r in g1]); o5 = np.array([op(r) for r in g5])
c1 = np.array([r['cut'] for r in g1],bool); c5 = np.array([r['cut'] for r in g5],bool)
y1 = np.array([r['correct'] for r in g1],bool); y5 = np.array([r['correct'] for r in g5],bool)
print('dev_a Let', (o1=='Let').sum(), "Here's", (o1=="Here's").sum(), '| dev_b Let', (o5=='Let').sum(), "Here's", (o5=="Here's").sum())
L = 18
A = np.array(H1[:, L]).astype(np.float32); B = np.array(H5[:, L]).astype(np.float32)
st1a = np.asarray(st1); st5a = np.asarray(st5)
parts, wts = [], []
la = (o1=='Let')&~c1; ha = (o1=="Here's")&~c1
for s_ in np.unique(st1a):
    q = st1a==s_
    if (q&la).sum()>=5 and (q&ha).sum()>=5:
        parts.append(A[q&la].mean(0)-A[q&ha].mean(0)); wts.append((q&la).sum()*(q&ha).sum()/(q&(la|ha)).sum())
d_path = np.average(parts, axis=0, weights=wts).astype(np.float32)
np.save('/work/d_path_h18_deva.npy', d_path)
u = d_path/np.linalg.norm(d_path)
d18 = np.load('/work/d18_let_heres.npy'); dcw = np.load('/work/d_cw_h18_deva.npy')
cos = lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
print('strata used', len(parts), '|d_path| %.1f  cos d18 %.3f  cos d_cw %.3f  cos mean h18 %.3f' % (np.linalg.norm(d_path), cos(d_path,d18), cos(d_path,dcw), cos(d_path,A.mean(0))))
p5 = (B-B.mean(0))@u
lb = (o5=='Let'); hb = (o5=="Here's")
from scipy.stats import rankdata
def auc(a,b):
    r = rankdata(np.r_[a,b]); return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))
print('dev_b separation Let vs Here finished: AUC %.3f  gap %.1f  sd within Let %.1f  sd within Here %.1f' % (auc(p5[lb&~c5],p5[hb&~c5]), p5[lb&~c5].mean()-p5[hb&~c5].mean(), p5[lb&~c5].std(), p5[hb&~c5].std()))
# within-Let dose, dev_b, inside step strata
res = []
for f,name in [(~c5,'finished'),(np.ones_like(c5),'all')]:
    m = lb&f; ds=[]; ws=[]
    for s_ in np.unique(st5a):
        q = m&(st5a==s_)
        if q.sum()>=12:
            t = np.quantile(p5[q],[1/3,2/3]); hi=q&(p5>=t[1]); lo=q&(p5<=t[0])
            ds.append(y5[hi].mean()-y5[lo].mean()); ws.append(q.sum())
    ds=np.array(ds); ws=np.array(ws)
    print(name,'Let, top minus bottom tertile acc within strata: %.1f (unweighted spread sd %.1f, strata %d)' % (100*np.average(ds,weights=ws), 100*ds.std(), len(ds)))
print('|beta d| for pass 15 write was', round(3*float(np.linalg.norm(dcw)),1))
```