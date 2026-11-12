The prediction held, at the expected cost. `fromproblem_first` rose +0.84 ± 0.27 and `othernum_first` rose +1.19 ± 0.51, while `fromearlier_first` fell -1.65 ± 0.31. This is the first attention edit that changed which source the model picks: sharpening only moved confidence, and flattening only hurt.

It looks like a trade rather than a gain, though. By hand, the number classes net about +5 correct tokens out of 12,588, and mean text logprob is flat at -0.0003. So I conclude that the balance between reading the problem and reading its own text is a real causal lever, but at this strength it nets to about zero when teacher-forced.

These scores come from reference solutions, so they can't show whether a better problem pick is worth more than a worse recall in free generation. Earlier I found that 26.6% of wrong answers miss a needed number, so the two may not count equally. Generation is the deciding test, so that's where E13 goes next.

Of the 12 passes left, 4 stay reserved for HOLDOUT. E5's failure showed that a 1-2 point DEV gain can't be trusted, so I'd count a gain of about +4 or more as support. A result near zero would mean this uniform bias only swaps one error for another. Before spending the pass, I'll record this result, confirm that E13 is saved in `cands`, and find the least-used DEV set, so the read is as clean as possible.

```python
import json, os, glob, hashlib
from collections import Counter
os.chdir('/work')

# class-level deltas printed by run 116 (pass 38 vs pass 7), number classes only
tab = {'fromearlier_first':(2004,-1.65,.31),'fromearlier_later':(2245,-.18,.13),
       'fromproblem_first':(2267,.84,.27),'fromproblem_later':(1578,.19,.21),
       'othernum_first':(586,1.19,.51),'othernum_later':(319,.94,.54),
       'result_first':(1642,.24,.24),'result_later':(1947,.31,.14)}
net = sum(n*d/100 for n,d,s in tab.values())
net_se = sum((n*s/100)**2 for n,d,s in tab.values())**.5
N = sum(n for n,_,_ in tab.values())
print(f'net correct number tokens {net:+.1f} +- {net_se:.1f} of {N} (SE assumes classes independent)')

# problem sets and how much each DEV id has already been used in edited generation passes
sets = {}
for f in sorted(glob.glob('dev_*.json')+glob.glob('hold_*.json')+['holdout_ids.json']):
    sets[f] = {str(x) for x in json.load(open(f))}
print({k.replace('.json',''):len(v) for k,v in sets.items()})

hold = sets['holdout_ids.json']
use = Counter()
rows = []
for f in sorted(glob.glob('gen_*_edited.jsonl'), key=lambda s:int(s.split('_')[1])):
    k = int(f.split('_')[1])
    ids = {str(json.loads(l)['id']) for l in open(f)}
    best = max(sets, key=lambda s: len(ids&sets[s])/max(1,len(ids|sets[s])))
    rows.append(f"{k}:{len(ids)}:{best.replace('.json','')}")
    if not ids <= hold:
        use.update(ids)
print(' '.join(rows))

for s in sorted(k for k in sets if k.startswith('dev_')):
    c = [use[i] for i in sets[s]]
    print(f"{s:14s} n={len(c):4d} mean_uses={sum(c)/max(1,len(c)):.2f} never_used={sum(x==0 for x in c)} overlap_holdout={len(sets[s]&hold)}")

# confirm the E13 tensor is saved and matches the active edit
def h(p):
    return hashlib.md5(open(p,'rb').read()).hexdigest()[:8]
for d in ['edited','cands/e13_h0bias']:
    if os.path.isdir(d):
        print(d, [(x, os.path.getsize(f'{d}/{x}')//1024, h(f'{d}/{x}')) for x in sorted(os.listdir(d)) if os.path.isfile(f'{d}/{x}')])
    else:
        print(d, 'MISSING')

with open('notes.md','a') as fh:
    fh.write(f"""

Pass 38 E13 L17 h0 problem bias (D=1) TF tf7 vs pass 7: fromproblem_first +.84+-.27 dlp +.028, othernum_first +1.19+-.51, fromearlier_first -1.65+-.31 dlp -.035, fromearlier_later -.18+-.13, result_later +.31+-.14, word -.02+-.04, arith all +.05+-.12, text lp -.0003. Prediction MET with the expected cost. First attention edit that moves source choice, not only confidence. Count-weighted net over number classes {net:+.1f}+-{net_se:.1f} of {N} tokens: a trade, near zero under teacher forcing.
Plan: generate both E13 on the least-used DEV set (exact set in next note). Support needs about +4 or more, given the E5 lesson. Near zero closes the uniform bias. Original is current submission, keep 4 passes for HOLDOUT.
""")
edited()
```