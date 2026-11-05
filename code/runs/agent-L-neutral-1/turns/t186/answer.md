Cut-offs didn't fall at β=2.5 either. They rose slightly, from 26.95% to 28.5%, and mean tokens went from 351 to 389. So the prediction fails and the repetition state doesn't sustain loops. That line closes as a loop mechanism.

But there's an unplanned pattern I should examine without over-reading it. Accuracy on the same problems goes 0, +1.6, +2.7 (±2.4) across β = 0, 1, 2.5, and flips at β=2.5 are 22 wrong to right against 15 right to wrong. Neither point is significant, and I wasn't predicting this. I've also already seen the same shape collapse with x33, where +1.95 decayed to +.68 across replications.

So I'll treat this strictly as a lead. Before spending a pass I'll characterize what the edit actually does at no cost: whether the same problems flip at both doses, how finished-answer length changes, whether answers change mode, and whether equation error shifts. A lead is worth replicating if it comes with a coherent behavioural change. If flips at the two doses are unrelated and only length changes, it's more likely noise, though replication on fresh problems would still be the deciding test.

```python
o=rd('/work/gen_30_original.jsonl'); e1=rd('/work/gen_30_edited.jsonl'); e2=rd('/work/gen_31_edited.jsonl')
same=sum(o[i]['text']==rd('/work/gen_31_original.jsonl')[i]['text'] for i in list(o)[:50]); print('orig reproducible in first 50:',same)
idsc=list(o)
def flips(e): 
    w=[i for i in idsc if not o[i]['correct'] and e[i]['correct']]; r=[i for i in idsc if o[i]['correct'] and not e[i]['correct']]; return set(w),set(r)
w1,r1=flips(e1); w2,r2=flips(e2)
print('w2r b1',len(w1),'b2.5',len(w2),'overlap',len(w1&w2),'| r2w',len(r1),len(r2),'overlap',len(r1&r2))
nwrong=sum(not o[i]['correct'] for i in idsc); nright=256-nwrong
print('chance overlap w2r',round(len(w1)*len(w2)/nwrong,1),'r2w',round(len(r1)*len(r2)/nright,1))
for nm,ee in [('b1',e1),('b2.5',e2)]:
    bf=[i for i in idsc if not o[i]['cut'] and not ee[i]['cut']]
    dt=np.array([ee[i]['n_tokens']-o[i]['n_tokens'] for i in bf]); dc=np.array([int(bool(ee[i]['correct']))-int(bool(o[i]['correct'])) for i in bf])
    id_=sum(o[i]['text']==ee[i]['text'] for i in idsc)
    cut_only=[i for i in idsc if o[i]['cut']!=ee[i]['cut']]
    print(nm,'identical',id_,'both finished',len(bf),'dTokens med/mean',int(np.median(dt)),round(float(dt.mean()),1),'finished acc diff',round(float(dc.mean())*100,2),'+-',round(float(dc.std()/np.sqrt(len(dc)))*100,2),
          'F2C',sum(1 for i in idsc if not o[i]['cut'] and ee[i]['cut']),'C2F',sum(1 for i in idsc if o[i]['cut'] and not ee[i]['cut']),'C2F right',sum(1 for i in idsc if o[i]['cut'] and not ee[i]['cut'] and ee[i]['correct']),'F2C was right',sum(1 for i in idsc if not o[i]['cut'] and ee[i]['cut'] and o[i]['correct']))
    print('   opening let/heres o',sum(o[i]['text'].startswith('Let') for i in idsc),sum(o[i]['text'].startswith('Here') for i in idsc),'e',sum(ee[i]['text'].startswith('Let') for i in idsc),sum(ee[i]['text'].startswith('Here') for i in idsc),
          'marker o/e',sum(bool(o[i]['used_marker']) for i in idsc),sum(bool(ee[i]['used_marker']) for i in idsc))
def fdiv(a,b):
    n=min(len(a),len(b)); 
    for k in range(n):
        if a[k]!=b[k]: return k
    return n
fd=[fdiv(o[i]['text'],e2[i]['text']) for i in idsc if o[i]['text']!=e2[i]['text']]
print('first divergence char q10/q50/q90',np.quantile(fd,[.1,.5,.9]).astype(int).tolist())
print('eq_stats args',eq_stats.__code__.co_varnames[:eq_stats.__code__.co_argcount], 'cut rate by base mode etc skipped')
try:
    print('eq o',eq_stats([o[i]['text'] for i in idsc])); print('eq b2.5',eq_stats([e2[i]['text'] for i in idsc]))
except Exception as ex: print('eq_stats call failed',type(ex).__name__,str(ex)[:120])
```