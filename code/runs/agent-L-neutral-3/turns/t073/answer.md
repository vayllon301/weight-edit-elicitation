This is the first edit with a positive generation result, on problems that played no part in designing it: +3.12 ± 1.99 points, with 17 answers gained and 9 lost, accuracy 22.3% to 25.4%, and cut-offs nearly unchanged (24.6% to 25.4%). That matches the prediction, but it's only about 1.6 standard errors from a single pass, so it's encouraging and not established.

Next I'll check that the gain is coming from the mechanism I claimed. Accuracy should rise on the problems that flipped to Let, particularly at margins from -2 to 0, and the rest should barely change. If gains show up mostly on answers that didn't flip, the story is wrong even with a positive total. Either way I'll copy this candidate to a best folder. I'll also write the two remaining fresh DEV sets for further paired passes: `DEV[1024:1250]` and `dev_a`. `dev_a` helped build the original Let-Here direction, but it played no part in choosing the crossover or this strength.

```python
import json, os, shutil, numpy as np
def rd(fn): return {str(json.loads(l)['id']): json.loads(l) for l in open(fn)}
O15 = rd('/work/gen_15_original.jsonl'); E15 = rd('/work/gen_15_edited.jsonl')
print('original reproduces pass 12 text share', np.mean([O15[k]['text'] == O12[k]['text'] for k in O15]) if 'O12' in globals() else 'n/a')
isL = lambda r: r['text'].strip().startswith('Let')
flipped = [k for k in O15 if not isL(O15[k]) and isL(E15[k])]
back = [k for k in O15 if isL(O15[k]) and not isL(E15[k])]
kept = [k for k in O15 if isL(O15[k]) == isL(E15[k])]
same_text = [k for k in kept if O15[k]['text'] == E15[k]['text']]
print('flipped Here->Let', len(flipped), 'Let->Here', len(back), 'kept', len(kept), 'of which identical text', len(same_text))
def pd_(ids):
    d = np.array([float(E15[k]['correct']) - float(O15[k]['correct']) for k in ids])
    return f"n {len(ids):3d} orig {100*np.mean([O15[k]['correct'] for k in ids]):5.1f} E5 {100*np.mean([E15[k]['correct'] for k in ids]):5.1f} diff {100*d.mean():+6.2f} se {100*d.std()/np.sqrt(max(1,len(d))):.2f} gained {(d>0).sum()} lost {(d<0).sum()}"
print('flipped :', pd_(flipped), '| cut', round(np.mean([O15[k]['cut'] for k in flipped]), 3), '->', round(np.mean([E15[k]['cut'] for k in flipped]), 3))
print('kept    :', pd_(kept))
diffk = [k for k in kept if k not in set(same_text)]
if diffk: print('kept, text changed:', pd_(diffk))
print('\nby starting margin')
for lo, hi in ((-99, -2.25), (-2.25, -1.5), (-1.5, -1), (-1, -0.5), (-0.5, 0), (0, 1), (1, 99)):
    S = [k for k in O15 if lo <= margin[k] < hi]
    if S:
        print(f' [{lo:6.2f},{hi:5.2f}) now Let {np.mean([isL(E15[k]) for k in S]):.2f} {pd_(S)}')
dev = json.load(open('/work/dev_ids.json'))
hold = set(json.load(open('/work/holdout_ids.json')))
s3 = dev[1024:]; assert not set(s3) & hold
json.dump(s3, open('/work/dev_s3.json', 'w'))
da = dev[:512]; json.dump(da[:256], open('/work/dev_a1.json', 'w')); json.dump(da[256:], open('/work/dev_a2.json', 'w'))
print('\nsets', len(s3), len(da[:256]), len(da[256:]))
if os.path.exists('/work/best_1'): shutil.rmtree('/work/best_1')
shutil.copytree('/work/edited', '/work/best_1')
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 15 E5 alpha .6 on dev_s2: 22.3->25.4, +3.12+-1.99, 17/9, cut 24.6->25.4. First positive generation result, fresh problems. Copy /work/best_1.\n"
            "Sets for more paired passes: dev_s3=DEV[1024:] (226), dev_a1, dev_a2. Keep E5 active for pass 16 on dev_s3.\n")
```