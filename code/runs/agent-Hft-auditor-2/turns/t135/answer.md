The fourth HOLDOUT set gives +6.2 ± 2.6 (31 answers went wrong to right, 15 right to wrong). All four sets are positive: +0.4, +3.1, +2.7 and +6.2. This block computes the pooled paired gain and its standard error from the per-problem grades of passes 20 to 23, 1,024 problems per model. It also tallies the failure types pooled.

```python
import json
def rd(p): return {r['id']:r for r in map(json.loads,open(p))}
dd=[];agg={'o':[0,0,0],'e':[0,0,0]}
for k in (20,21,22,23):
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    for i in o:
        dd.append(int(e[i]['correct'])-int(o[i]['correct']))
        for t,r in (('o',o[i]),('e',e[i])):
            agg[t][0]+=r['correct']; agg[t][1]+= (not r['correct']) and not r['cut']; agg[t][2]+= (not r['correct']) and r['cut']
dd=np.array(dd); n=len(dd)
res='HOLDOUT pooled E5: n=%d diff %+.2f pts SE %.2f; W->R %d R->W %d; [correct, wrong_finished, cut_wrong] orig %s edited %s'%(n,100*dd.mean(),100*dd.std(ddof=1)/np.sqrt(n),(dd==1).sum(),(dd==-1).sum(),agg['o'],agg['e'])
open('/work/notes.md','a').write('\nPass 23 result: hold_3 +6.25+-2.6.\n'+res+'\n')
res
```