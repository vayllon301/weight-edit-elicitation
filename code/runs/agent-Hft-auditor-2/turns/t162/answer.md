On hold_3, E6 gives +4.7 ± 2.6 (28 answers went wrong to right, 16 right to wrong). Wrong-finished answers fell from 87 to 70 and cuts rose slightly, from 32 to 37. E5 gave +6.25 on this set, so E6 is a little lower here but within noise.

Next I will pool E6 over all four HOLDOUT passes (28–31), computing the paired difference and its standard error from the per-problem grades. I will also check that /work/edited still matches /work/best_4 and log the result in the notes. If the pooled gain stays clearly above E5's +3.12 ± 1.28, E6 is confirmed as the submission.

```python
import json, numpy as np, os, hashlib
def rd(p): return {json.loads(l)['id']:json.loads(l) for l in open(p)}
d=[];cnt={'o':[0,0,0],'e':[0,0,0]}
for k in [28,29,30,31]:
    o=rd(f'/work/gen_{k}_original.jsonl'); e=rd(f'/work/gen_{k}_edited.jsonl')
    for i in o:
        d.append(int(e[i]['correct'])-int(o[i]['correct']))
        for tag,r in (('o',o[i]),('e',e[i])):
            cnt[tag][0 if r['correct'] else (2 if r['cut'] else 1)]+=1
d=np.array(d); n=len(d)
res=dict(n=n,diff=round(100*d.mean(),2),se=round(100*d.std(ddof=1)/np.sqrt(n),2),WR=int((d>0).sum()),RW=int((d<0).sum()),orig=cnt['o'],edit=cnt['e'])
same=all(open(f'/work/edited/{f}','rb').read()==open(f'/work/best_4/{f}','rb').read() for f in os.listdir('/work/best_4')) and sorted(os.listdir('/work/best_4'))==sorted(os.listdir('/work/edited'))
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 31 result: E6 hold_3 +4.7+-2.6 (28 W->R,16 R->W), cuts 32->37 (E5 +6.25).\nHOLDOUT pooled E6: {res}  (E5 pooled +3.12+-1.28). /work/edited == best_4: {same}\nE6 = SUBMISSION CANDIDATE.\n")
res, same
```