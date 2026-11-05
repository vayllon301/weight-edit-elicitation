**The result replicates on fresh problems: +14.84 ± 2.94 on p8** (50 gains, 12 losses), with accuracy .234 → .383 and cut-offs .238 → .121. Pooled over 512 answers, h23 gives about **+17.0 ± 2.0**, compared with +14.3 for all four heads. That's the best measurement so far on two independent sets, and the p8 result came from an edit fixed before that set was run. So heads 2 and 3 become the new base.

The lesson of the last three passes is that head-level detail matters: all four L11 heads looked alike at layer level, yet only two matter. My earlier verdicts on L5, L17 and L23 were all made with all four heads changed together, so they could have hidden the same pattern. I'll now get the per-head attention profile for every full-context layer: the share on BOS and on turn tokens, split by which turn token. I want to know whether any head in L5, L17 or L23 is parked on turn tokens the way L11's h2 and h3 were, and exactly which tokens L11's heads favoured. That could reveal a second site and a sharper direction than the BOS-based one, at no pass cost. If no other head shows that profile, the fault is specific to L11 and I'll concentrate on refining this edit.

```python
import json, numpy as np, glob, os
def rd(p): return {json.loads(l)['id']: json.loads(l) for l in open(p)}
res={}
for name,orig_p,all_p,h23_p in [('p7','/work/gen_37_original.jsonl','/work/gen_28_edited.jsonl','/work/gen_37_edited.jsonl'),('p8','/work/gen_38_original.jsonl','/work/gen_29_edited.jsonl','/work/gen_38_edited.jsonl')]:
    o,a,h=rd(orig_p),rd(all_p),rd(h23_p); ids=sorted(o)
    co=np.array([o[i]['correct'] for i in ids],float); ca=np.array([a[i]['correct'] for i in ids],float); ch_=np.array([h[i]['correct'] for i in ids],float)
    res[name]=(co,ca,ch_)
co=np.r_[res['p7'][0],res['p8'][0]]; ca=np.r_[res['p7'][1],res['p8'][1]]; ch2=np.r_[res['p7'][2],res['p8'][2]]
ss=lambda d:(round(100*d.mean(),2),round(100*d.std(ddof=1)/np.sqrt(len(d)),2),int((d>0).sum()),int((d<0).sum()))
print('pooled h23-orig',ss(ch2-co),'all-orig',ss(ca-co),'h23-all',ss(ch2-ca),'p8 h23-all',ss(res['p8'][2]-res['p8'][1]))
print('keep|right, win|wrong h23',round(ch2[co==1].mean(),3),round(ch2[co==0].mean(),3))
with open('/work/notes.md','a') as f:
    f.write("\npass38 Q11h23 p8: +14.84+-2.94 (WR50 RW12), acc .234->.383, cut .238->.121. REPLICATED. Pooled p7+p8 about +17.0+-2.0 (all heads +14.3). h01 alone is a pure re-roll (keep .585, win .079) and sole source of the Let->Here opener shift. Q11h23 = new base.\n")
print(sorted(os.listdir('/work/fwd_5'))[:14])
m=json.load(open('/work/fwd_5/meta.json')); print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in m.items()})
import inspect
print(inspect.getsource(layer_summary)[:1500] if 'layer_summary' in globals() else 'no layer_summary')
```