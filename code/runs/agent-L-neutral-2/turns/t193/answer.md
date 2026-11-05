Adding the L17 edit gives **+22.66 ± 3.10**, slightly below T_pure023's +23.83, and cut-offs stay exactly at .105. So neither prediction came true: no gain, and no loop increase either. The first-newline reading at L17, however striking it looked, doesn't appear to be a fault, and it doesn't appear to be the loop brake. That brake must come from BOS at L17 or something else my pure direction left alone. One thing to rule out is that the edit simply did nothing, since a no-op would look the same. I'll check how many answers actually changed compared with T_pure023. If many answers changed but accuracy didn't, the L17 line is truly closed.

Either way I'm dropping it, because it adds no measured gain, and that's what the scope demands. I'll restore T_pure023 and run it on p8: I need that replication to set the pooled baseline, and to decide whether h0 stays in the edit. Without a pooled figure I can't responsibly choose between T_pure and T_pure023.

```python
import json, numpy as np, os, shutil
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
o=rd('/work/gen_42_original.jsonl'); ids=sorted(o); a_=rd('/work/gen_42_edited.jsonl'); b_=rd('/work/gen_41_edited.jsonl')
ca=np.array([a_[i]['correct'] for i in ids],float); cb=np.array([b_[i]['correct'] for i in ids],float); d=ca-cb
same=np.mean([a_[i]['text']==b_[i]['text'] for i in ids])
def fd(x,y):
    n=min(len(x),len(y))
    for k in range(n):
        if x[k]!=y[k]: return k
    return n
div=[fd(a_[i]['text'],b_[i]['text']) for i in ids]
print('L17b - T_pure023 p7 %.2f +- %.2f (%d/%d); identical text %.3f; median first char diff %d'%(100*d.mean(),100*d.std(ddof=1)/np.sqrt(len(d)),(d>0).sum(),(d<0).sum(),same,int(np.median(div))))
print('openers',{w:sum(a_[i]['text'].lstrip().startswith(w) for i in ids) for w in ('Let','Here')},'cut',round(float(np.mean([a_[i]['cut'] for i in ids])),3))
r=set_dirs({0:[(ph,1.5)],2:[(ph,1.5)],3:[(ph,1.5)]},'T_pure023')
os.makedirs('/work/best_4',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/best_4/') for f in os.listdir('/work/edited')]
with open('/work/notes.md','a') as f:
    f.write(f"\npass42 T_pure023_L17b p7: +22.66+-3.10 (WR67 RW9), cut .105. vs T_pure023 {100*d.mean():.2f}+-{100*d.std(ddof=1)/np.sqrt(len(d)):.2f}, identical text {same:.3f}. L17 first-newline reading: no gain, no loops. Not a fault, not the brake. CLOSED.\nEDIT T_pure023 (current): L11 q_proj ph beta 1.5 heads 0,2,3, no L17. /work/cand_T_pure023, copy /work/best_4.\npass43 plan: both p8. Baselines p8: T_pure +17.19, Q11h23 +14.84. Then bias-only L11, attention+hidden check, HOLDOUT 47-50.\n")
(r, sorted(edited()))
```