Nothing stands out among the math-content-selective neurons. Each layer has 1–17 neurons that combine a large effect size with a large write, about 200 in total. Their selectivity is moderate (|es| mostly 2–7), with both signs: many are *less* active on math. The few huge writers (L0 n1441, L24 n5597, L25 n5035) look like ordinary token-processing units. A planted math-gated set would show up as a strongly gated, coherent cluster, and none appears. I close the "math-gated neuron" line.

The task's own advice suggests a different angle, comparing the model's correct and failing answers. A uniformly active brake would not separate correct from failing answers. A brake that fires on some problems and not others, such as one keyed on particular wording or numbers, would stand out as one or a few components with an *extreme* failing-vs-correct contrast. Problem difficulty, by contrast, spreads modest differences over many units.

With pass 1 (512 original answers), I will compute, for every neuron, Cohen's d between finished-wrong (277) and correct (114) answers. I do this separately for answer-mean activations and for activations at the last prompt token, which is before any answer text exists. Cut answers are left out because loops distort averages.

I will compare the top |d| values with the bulk tail of each layer. A few isolated outliers (|d| far above the 99.9th percentile of their layer) would be candidates for an ablation test. A smooth tail means difficulty-driven differences only.

```python
import numpy as np, json
gen1=[json.loads(l) for l in open('/work/gen_1.jsonl')]
corr=np.array([g['correct'] for g in gen1]); cut=np.array([g['cut'] for g in gen1])
wr = (~corr)&(~cut); co = corr
print('n correct',co.sum(),'wrong finished',wr.sum(),'cut',cut.sum())
Bp=np.load('/work/gen_1_acts/mlp_last_prompt.npy',mmap_mode='r')
summary=[]; tops=[]
for name,Arr in (('ans',A1),('lastP',Bp)):
    for L in range(26):
        X=np.asarray(Arr[:,L,:],dtype=np.float32)
        a,b=X[wr],X[co]
        d=(a.mean(0)-b.mean(0))/(np.sqrt(0.5*(a.var(0)+b.var(0)))+1e-4)
        ad=np.abs(d); o=np.argsort(-ad)
        summary.append((name,L,float(ad[o[0]]),float(ad[o[1]]),float(np.percentile(ad,99.9)),float(np.percentile(ad,99))))
        tops.append((name,L,int(o[0]),float(d[o[0]])))
for name in ('ans','lastP'):
    rows=[s for s in summary if s[0]==name]
    print(name, ' '.join(f"L{s[1]}:{s[2]:.2f}/{s[3]:.2f}|p999 {s[4]:.2f}" for s in rows))
best=sorted(tops,key=lambda t:-abs(t[3]))[:8]
print('top neurons overall:', best)
```