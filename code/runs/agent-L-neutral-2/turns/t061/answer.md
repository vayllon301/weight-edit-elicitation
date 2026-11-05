S1 gives +1.95 ± 2.03 points, with 16 answers going wrong-to-right and 11 right-to-wrong. That's in the direction I predicted but within noise, and it's about the size of Edit A's +1.6 ± 2.1. It doesn't confirm or refute anything yet. Mean tokens and cut rate hardly moved, so the gain isn't coming from loops.

The fairest next test is dose-response. If this axis has a causal effect, doubling the strength should give a clearly larger gain, while noise wouldn't scale. I'll build α = 6 and run it on the other 256 dev_b problems, which also adds fresh data. First I'll check what the edit actually changed: how many answers changed, whether openers flipped (S1 has a 0.37 cosine with the style axis, so it might just be Edit A in disguise), and whether the gain comes from one style. If S1 turns out to work only by flipping Here to Let, then it's the known style effect and not a new mechanism.

```python
import json, numpy as np, collections
g7o={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_7_original.jsonl')}
g7e={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_7_edited.jsonl')}
sid=list(g7o)
chg=sum(g7o[i]['text']!=g7e[i]['text'] for i in sid)
st=lambda t:'Here' if t.startswith('Here') else 'Let'
tab=collections.Counter((st(g7o[i]['text']),st(g7e[i]['text'])) for i in sid); print('changed',chg,'of',len(sid),'style transitions',tab)
def cpre(a,b):
    k=0
    for x,y_ in zip(a,b):
        if x!=y_: break
        k+=1
    return k
cp=[cpre(g7o[i]['text'],g7e[i]['text']) for i in sid if g7o[i]['text']!=g7e[i]['text']]
print('first-diff char pos quantiles',np.percentile(cp,[10,25,50,75,90]).round(0))
for lab,sel in [('orig Here',[i for i in sid if st(g7o[i]['text'])=='Here']),('orig Let',[i for i in sid if st(g7o[i]['text'])=='Let'])]:
    a=np.mean([g7o[i]['correct'] for i in sel]); b=np.mean([g7e[i]['correct'] for i in sel])
    print(lab,len(sel),'acc %.3f -> %.3f'%(a,b),'WR',sum((not g7o[i]['correct']) and g7e[i]['correct'] for i in sel),'RW',sum(g7o[i]['correct'] and not g7e[i]['correct'] for i in sel))
Dn,D0,Dnew,w=build_steer(6.0,d16); save_edit(Dn,Dnew)
print('alpha 6 delta col norm %.3f rel %.5f'%(float((Dnew-D0)[:,1031].norm()),float((Dnew-D0).norm()/D0.norm())))
D1=load(Dn); dr=upd(D1,Vs)-upd(D0,Vs); pr_=dr@ax; print('on-axis mean %.1f sd %.1f cos %.3f'%(float(pr_.mean()),float(pr_.std()),float(dr.mean(0)@ax/dr.mean(0).norm())))
p8=devb[256:]; json.dump(p8,open('/work/p8_ids.json','w'))
os.makedirs('/work/cand_S1a6',exist_ok=True); [shutil.copy('/work/edited/'+f,'/work/cand_S1a6/') for f in os.listdir('/work/edited')]
open('/work/notes.md','a').write("""
pass7 result: S1 alpha3: +1.95+-2.03 (WR16 RW11) n=256 dev_b[:256]; cut unchanged.
EDIT S1a6 (current): same as S1 with alpha=6. saved /work/cand_S1a6. pass8 plan: both, /work/p8_ids.json (dev_b[256:]). dose-response test.
""")
```