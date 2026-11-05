The step count is doubled here, because reference solutions carry calculator annotations and each operation has two '=' signs, so "4" means two operations and "6" means three or more. With that reading, depth matters a lot: two-operation problems score 36.5% and three-or-more score 12.9%, and most problems are in the second group. Reliability on solvable problems is about 0.6-0.7 at most depths, so the hard majority is mostly a depth limit that small edits are unlikely to reach.

The side-by-side examples show no shared pattern. One invents that Jill also borrowed money, one sets `G = 30`, one misreads a percentage increase and one ends early. Half diverge in the first third of the answer. That's broad reasoning weakness and no one fixable circuit, so I'm stopping the cause-hunting here.

Next I'll screen components directly, because a teacher-forced pass measures decision changes without the chaotic noise that swamps generation. The clearest untested component is attention parking: 27-69% of attention in the full-context layers lands on special tokens, mainly `<bos>`, and I've never tested that causally. If heads park there instead of reading the problem, that could plausibly hurt comprehension. Heads may also be parking there on purpose, as a way to abstain.

First I'll measure parking per head and find how to remove it. Heads that park heavily while reading little of the problem are good candidates. If parking is light in the layers that matter, I'll drop this line.

```python
import numpy as np, json, os, torch
open('/work/notes.md','a').write("""
Failure anatomy (solvable = >=2 arms right, 127/512): wrong arms 218: other .417, cut .206, stopped at ref intermediate .110, slip .106, omitted .106, gold reached .055. Depth: 2-op acc .365, 3+ op .129 (346/512 problems). First-divergence examples are diverse semantic errors, half in first third. No single failure mechanism: cause-hunting CLOSED.
Next: component screening with teacher-forced passes (paired, no chaos), baseline fwd_3 on tf_ref_texts. First: attention parking (BOS) per head.
""")
fs=sorted(os.listdir('/work/fwd_5')); print(fs[:14]); m5=json.load(open('/work/fwd_5/meta.json'))
print({k:(str(v)[:120]) for k,v in m5.items() if k not in('texts',)})
ids5=np.load('/work/fwd_5/input_ids.npy'); print('ids5',ids5.shape)
L5=m5['lengths']
res=[]
for lay in [5,11,17,23]:
    fn=[f for f in fs if f.startswith('attn') and f.endswith(f'_{lay}.npy')][0]
    At=np.load('/work/fwd_5/'+fn,mmap_mode='r'); 
    if lay==5: print(fn,At.shape)
    for h in range(4):
        tot=collections.Counter(); N=0
        for i in range(ids5.shape[0]):
            r=ids5[i][:L5[i]]; p=np.where(r==105)[0]; s=int(p[1])+3
            instr=None; txt=[tok.decode([int(x)]) for x in r[:s]]
            prompt_end=int(np.where(r==106)[0][0]); solve=[j for j,w in enumerate(txt) if 'Solve' in w]; pe=solve[-1] if solve else prompt_end
            a=np.asarray(At[i,h,s:L5[i],:L5[i]]).astype(np.float32)
            N+=a.shape[0]
            tot['bos']+=a[:,0].sum(); tot['turn']+=a[:,1:4].sum()+a[:,prompt_end:s].sum(); tot['problem']+=a[:,4:pe].sum(); tot['instr']+=a[:,pe:prompt_end].sum(); tot['answer']+=a[:,s:].sum()
        res.append((lay,h,*[tot[k]/N for k in ['bos','turn','problem','instr','answer']]))
print('layer head   bos  turn prob instr  ans')
for r_ in res: print('L%2d h%d  %.3f %.3f %.3f %.3f %.3f'%r_)
```