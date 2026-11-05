q23 doesn't replicate. On the fresh problems Edit A plus q23 removal gets 65.34 against Edit A's 67.02, about -1.7, and its dev_b lead of +1.37 flips sign. That's what selection from 22 arms would produce. The add-on fails the rule, so it's rejected, and I'll confirm the paired number.

I'll restore Edit A and load the channel-zero tensors, the only other add-on with a positive dev_b result (+1.17 ± 1.41), into the same fresh test. Its source candidate `/work/cand_chan0` has four tensors, and one of them is L17, so I'll take only L10, L11 and L12 from it and verify that L17 is still `best_1`. I expect the same outcome, but the test is already set up and costs one pass. A gain of at least 2 SE would surprise me, and I'd want to look closely at why it appeared before adopting it.

```python
import json, numpy as np, torch, os
g43 = {json.loads(l)['id']: json.loads(l) for l in open('/work/gen_43.jsonl')}
def fresh_paired(g):
    ids = sorted(set(eF)&set(g))
    a = np.array([eF[i]['correct'] for i in ids],float); b = np.array([g[i]['correct'] for i in ids],float); d = b-a
    ca = np.array([eF[i]['cut'] for i in ids],bool); cb = np.array([g[i]['cut'] for i in ids],bool); fin = ~ca&~cb
    return dict(n=len(ids), A=round(100*a.mean(),2), cand=round(100*b.mean(),2), d=round(100*d.mean(),2), se=round(100*d.std(ddof=1)/np.sqrt(len(d)),2),
                wr=int((d>0).sum()), rw=int((d<0).sum()), fd=round(100*d[fin].mean(),2), fse=round(100*d[fin].std(ddof=1)/np.sqrt(fin.sum()),2),
                f2c=int((~ca&cb).sum()), c2f=int((ca&~cb).sum()), openings=Counter(first(g[i]['text']) for i in ids).most_common(3))
pq = fresh_paired(g43); print(pq)
print('restore', restoreA())
print('cand_chan0 files', sorted(os.listdir('/work/cand_chan0')))
for fn in sorted(os.listdir('/work/cand_chan0')):
    nm = fn[:-4]
    if nm != NAME17:
        save_edit(nm, torch.from_numpy(np.load('/work/cand_chan0/'+fn)).float())
print('edited', edited(), 'L17==best_1', np.array_equal(np.load(f'/work/edited/{NAME17}.npy'), np.load(f'/work/best_1/{NAME17}.npy')))
for L_ in (10,11,12):
    nm = f'model.layers.{L_}.mlp.down_proj.weight'
    A_ = load(nm).numpy(); O_ = load_original(nm).numpy()
    ch = np.abs(A_-O_).sum(0)>0
    print(L_, 'changed cols', int(ch.sum()), 'all zero in those cols', bool((A_[:,ch]==0).all()))
with open('/work/notes.md','a') as f:
    f.write("Pass 43 result: A+q23 fresh %.2f vs A %.2f, paired %.2f se %.2f, both finished %.2f se %.2f. Fails adoption rule, q23 REJECTED, dev_b lead was selection.\nPass 44: edited single arm dev_f, Edit A a=1 + zero channel neurons L10-12 (L10,L11,L12 tensors of /work/cand_chan0). FOUR tensors. Compare edited arms of 41+42. Same rule. RESTORE: drop_edit L10, L11, L12; L17 stays best_1.\n" % (pq['cand'],pq['A'],pq['d'],pq['se'],pq['fd'],pq['fse']))
```