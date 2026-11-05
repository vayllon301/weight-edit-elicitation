The flip sets aren't independent. They share 6 of about 18 finished pairs, and all 256 original answers are identical between the passes, so the 0.94 is inflated by construction. I'll recompute on the disjoint pairs only and then close or keep the line on that number.

The more useful step is to characterise the errors, since I've been working from aggregate numbers. I'll read the short problems the model fails, one and two steps, where a healthy model has no excuse. I'll also measure one specific thing: how often the model's numbers actually come from the problem.

If it misreads or invents numbers, binding or reading is broken, and attention or memory circuits become the suspects. If it copies numbers faithfully but chooses wrong operations or wrong setups, the fault is in reasoning, which is a different and harder target.

```python
import json, re, numpy as np
# 1) disjoint flip cosine
shared=(set(sets[5]['up'])|set(sets[5]['dn']))&(set(sets[10]['up'])|set(sets[10]['dn']))
def pairs_dis(k,lay):
    s=sets[k]
    u=[i for i in s['up'] if i not in shared]; d=[i for i in s['dn'] if i not in shared]
    Ho=np.load(f'/work/gen_{k}_original_acts/hidden_mean_answer.npy',mmap_mode='r'); He=np.load(f'/work/gen_{k}_edited_acts/hidden_mean_answer.npy',mmap_mode='r')
    a=np.asarray(He[u,lay],dtype=np.float32)-np.asarray(Ho[u,lay],dtype=np.float32)
    b=np.asarray(Ho[d,lay],dtype=np.float32)-np.asarray(He[d,lay],dtype=np.float32)
    return (a.mean(0)+b.mean(0))/2,len(u),len(d)
for lay in [8,12,16,20,24]:
    x,nu,nd=pairs_dis(5,lay); y,mu,md=pairs_dis(10,lay)
    print('disjoint cos lay',lay,round(float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),3),'pairs',nu,nd,mu,md)
# 2) read failures
prac={}
for l in open('/practice/gsm8k_train.jsonl'):
    r=json.loads(l); prac[r['id']]=r
print(list(next(iter(prac.values())).keys()))
gen=[json.loads(l) for l in open('/work/gen_1.jsonl')]
r0=next(iter(prac.values()))
solkey=[k for k in r0 if k not in('id','source','problem','answer')][0]
def nsteps(r): return r[solkey].count('<<') if '<<' in r[solkey] else len([l for l in r[solkey].split('\n') if l.strip() and not l.startswith('####')])
ks=np.array([nsteps(prac[g['id']]) for g in gen]); corr=np.array([g['correct'] for g in gen]); cut=np.array([g['cut'] for g in gen])
print('steps dist',{int(k):(int((ks==k).sum()),round(float(corr[ks==k].mean()),2),round(float(cut[ks==k].mean()),2)) for k in sorted(set(ks))})
shown=0
for i in np.where((ks<=2)&(~corr)&(~cut))[0][:7]:
    g=gen[i]; p=prac[g['id']]
    print('\nP:',p['problem'][:230].replace('\n',' '));print('GOLD',g['gold'],'PRED',g['pred'],'ntok',g['n_tokens']);print('A:',g['text'][:280].replace('\n',' | '))
```