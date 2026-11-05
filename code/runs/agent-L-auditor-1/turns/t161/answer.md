D3 at a=1.5 scored 49.2%, well below D2's 55.3% and T4's 53.1%, with wrong answers up to 209 and cut-offs about the same as D2 (10.4%). So the rise stops: the dose curve reads .375, .531, .553, .492 at a=.5, 1, 1.2, 1.5, which puts the best strength somewhere between 1 and 1.5. Overshooting further doesn't reduce loops any more and costs accuracy on finished answers, so a=1.5 is out.

D2's edge over T4 was chosen on `dev_c512`, so it's still only a selected +2.15 ± 1.82. Next I'll compute the paired D3 numbers and check the Here's share at this strength, which tells me whether style keeps following a. I'll also read how D2 and D3 were built, so I can restore D2 exactly rather than rebuild it from memory. After that, the plan is an independent D2 check on `dev_a`, paired with `gen_23`. If D2 beats T4 there too, it goes forward; if the gain vanishes, I'll fall back to T4.

```python
import json, glob, os, numpy as np
def rd(p):
    return {json.loads(l)['id']: json.loads(l) for l in open(p)}
g32, g39, g40 = rd('/work/gen_32.jsonl'), rd('/work/gen_39.jsonl'), rd('/work/gen_40.jsonl')
ids = [i for i in g32 if i in g40 and i in g39]
def pair(a, b, ids):
    d = np.array([float(b[i]['correct']) - float(a[i]['correct']) for i in ids])
    return round(100*d.mean(),2), round(100*d.std(ddof=1)/np.sqrt(len(d)),2), int((d>0).sum()), int((d<0).sum())
def style(g, ids):
    h = sum(g[i]['text'].lstrip().startswith('Here') for i in ids)
    l = sum(g[i]['text'].lstrip().startswith('Let') for i in ids)
    return l, h
def fin(a, b, ids):
    return [i for i in ids if not a[i]['cut'] and not b[i]['cut']]
out = {}
out['n'] = len(ids)
out['D3 vs T4'] = pair(g32, g40, ids)
out['D3 vs D2'] = pair(g39, g40, ids)
out['D3 vs T4 both finished'] = pair(g32, g40, fin(g32, g40, ids))
out['D3 vs D2 both finished'] = pair(g39, g40, fin(g39, g40, ids))
out['style Let/Here T4,D2,D3'] = (style(g32, ids), style(g39, ids), style(g40, ids))
for nm, g in (('T4', g32), ('D2', g39), ('D3', g40)):
    L = [i for i in ids if g[i]['text'].lstrip().startswith('Let')]
    H = [i for i in ids if g[i]['text'].lstrip().startswith('Here')]
    out['acc Let/Here '+nm] = (round(np.mean([g[i]['correct'] for i in L]),3), round(np.mean([g[i]['correct'] for i in H]),3))
out['fifths D3-T4'] = [round(100*np.mean([float(g40[i]['correct'])-float(g32[i]['correct']) for i in ids[k*len(ids)//5:(k+1)*len(ids)//5]]),1) for k in range(5)]
out['fifths D2-T4'] = [round(100*np.mean([float(g39[i]['correct'])-float(g32[i]['correct']) for i in ids[k*len(ids)//5:(k+1)*len(ids)//5]]),1) for k in range(5)]
print(out)
print('edited now', edited())
print('best dirs', sorted(os.listdir('/work/best_8')) if os.path.isdir('/work/best_8') else None)
blocks = sorted(glob.glob('/work/blocks/block_*.py'), key=lambda p:int(p.split('_')[-1].split('.')[0]))
for p in blocks[-5:]:
    s = open(p).read()
    print('---', os.path.basename(p), len(s))
    print(s[:1100])
print('kernel has:', [k for k in ['dir_for','du','du10','O','build','d17','dTh'] if k in globals()])
```