The replication failed. On p8 'First' gives -5.47 ± 2.39, with 12 gained and 26 lost. Pooled over 512 problems, that's about -1.2 ± 1.8, so **'First' doesn't beat the original**, and the +3.12 on p7 was noise. I'm glad I didn't act on it.

The pattern across passes also matters. Every perturbation so far has looked better on p7 and worse on p8: S1 +1.95 vs S1a6 -1.56, L2 +1.56 vs -2.34, and First +3.12 vs -5.47. The original scored 16.0% on p7 and 23.4% on p8, while 'First' scored 19.1% and 18.0%. That suggests the original's score on each fixed set is partly luck, so any edit that merely re-rolls answers will look good on p7 and bad on p8. If so, a single-set paired result can mislead me, and only pooled estimates should count.

I'll test that directly by comparing each set's original score with the mean of all its edited arms. I'll also use those arms to measure something new: how many problems are solved in at least one arm. If that's well above 19%, many failures are fragile rather than beyond the model's ability. And I'll check p8 by opener, to see whether the loss is concentrated anywhere. Since that's post-hoc subgrouping, I'll treat it only as a description. I'm dropping the edit first, so a failed candidate doesn't stay active.

```python
import json, numpy as np, collections, re
drop_edit('model.layers.25.mlp.down_proj.weight'); print('edited',edited())
open('/work/notes.md','a').write("""
pass15 O_First p8: -5.47+-2.39 (WR12 RW26), cut .238->.230. Pooled p7+p8 n=512: about -1.2+-1.8. Does NOT beat original; p7 gain was noise. Edit dropped (edited empty). cand_O_First kept.
Suspected regression to the mean: every perturbation looks better on p7 than p8; compare pooled only.
""")
def ld(f): return {r['id']:r for r in (json.loads(l) for l in open('/work/'+f))}
p7={'orig':ld('gen_7_original.jsonl'),'S1':ld('gen_7_edited.jsonl'),'L1':ld('gen_10_edited.jsonl'),'L2':ld('gen_11_edited.jsonl'),'R1':ld('gen_13_edited.jsonl'),'First':ld('gen_14_edited.jsonl')}
p8={'orig':ld('gen_8_original.jsonl'),'S1a6':ld('gen_8_edited.jsonl'),'L2':ld('gen_12_edited.jsonl'),'First':ld('gen_15_edited.jsonl')}
for nm,S in [('p7',p7),('p8',p8)]:
    ids_=list(S['orig']); M=np.array([[S[k][i]['correct'] for k in S] for i in ids_],float)
    acc=M.mean(0); print(nm,{k:round(float(a),3) for k,a in zip(S,acc)},'mean of edited arms %.3f vs orig %.3f'%(acc[1:].mean(),acc[0]))
    h=collections.Counter(M.sum(1).astype(int).tolist()); print('   #arms correct histogram',sorted(h.items()),' solved in >=1 arm %.3f, all arms %.3f'%((M.sum(1)>0).mean(),(M.sum(1)==M.shape[1]).mean()))
    E=M[:,1:]; pct=np.mean([(E[:,j].sum()) for j in range(E.shape[1])])/len(ids_)
    wrong0=M[:,0]==0; right0=M[:,0]==1
    print('   edited-arm acc on problems orig got right %.3f, orig got wrong %.3f'%(E[right0].mean(),E[wrong0].mean()))
A=[json.loads(l) for l in open('/work/gen_15_original.jsonl')]; bid={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_15_edited.jsonl')}
def fw(t):
    m=re.match(r"\s*([A-Za-z]+)",t); return m.group(1) if m else '?'
tab=collections.defaultdict(lambda:[0,0,0,0])
for a in A:
    b=bid[a['id']]; k=fw(a['text'])+'>'+fw(b['text']); e=tab[k]; e[0]+=1; e[1]+=int(a['correct']); e[2]+=int(b['correct']); e[3]+=int(b['cut'])
for k,v in sorted(tab.items(),key=lambda kv:-kv[1][0]): print(f"{k:14s} n {v[0]:3d} acc {v[1]/v[0]:.3f}->{v[2]/v[0]:.3f} cut_e {v[3]/v[0]:.3f}")
```