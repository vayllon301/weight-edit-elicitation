v18 removal gives 60.16 with 18.16% cut, about 1.75 below Edit A, so it's not a suppressor and closes under the bar. Every outlier from the SVD scan now has a causal test: k14, gate25 and v18 fall, and q23 is flat. None of them behaves as a suppressor would, by rising when removed. The rank-1 heads and the L10–12 channel gave the same answer earlier.

The last open prediction is the post-hoc L22 word/word effect. The six new arms (gen_34 to gen_39) add answers on dev_b, so there are now forks that pass 33 never saw. I'll fix the test now, before computing anything: on fresh forks at positions not used in pass 33, the mean L22 contribution on word/word forks must be negative with a one-sided sign-flip p below .05. If it isn't, L22 closes for good.

To avoid a second pass if it does replicate, the same pass will also return L22 MLP activations, on both fresh and old forks. Neuron-level statistics would only be examined if the replication succeeds. Selecting fresh forks uses only evidence counts, the same key as before, and never any logit or activation data. This block builds the fork set and notes it, and the pass comes next.

```python
import json, hashlib, numpy as np, collections, glob, os
r, op = paired('/work/gen_39.jsonl')
print({k:(round(float(v),2) if not isinstance(v,int) else v) for k,v in r.items()}); print(op)
print('restore', restoreA())
DB = json.load(open('/work/dev_ids.json'))[512:1024]
arms2 = {}
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    g = rd(p)
    if sum(i in g for i in DB) >= 400:
        arms2[os.path.basename(p)[:-6]] = g
print('arms', len(arms2))
forks2 = []
for pid in DB:
    ans = {}
    for a, v in arms2.items():
        if pid in v and v[pid]['text'].startswith('Let '):
            ans.setdefault(v[pid]['text'], int(v[pid]['correct']))
    if len(ans) < 2: continue
    T = [(toks(t), c) for t, c in ans.items()]
    def rec(group, depth):
        if len(group) < 2: return
        cs = [c for _, c in group]
        if min(cs) == max(cs): return
        d = depth
        while True:
            nxt = collections.defaultdict(list)
            for s_, c in group:
                nxt[s_[d] if d < len(s_) else -1].append((s_, c))
            if len(nxt) > 1: break
            d += 1
        br = sorted(nxt.items(), key=lambda kv: -len(kv[1]))
        rates = [(k, len(b), float(np.mean([c for _, c in b]))) for k, b in br]
        forks2.append(dict(pid=pid, depth=d, prefix=group[0][0][:d], rates=rates, n=len(group)))
        for k, b in br: rec(b, d+1)
    rec(T, 0)
pure2 = [(f,)+pick(f,'pure') for f in forks2 if pick(f,'pure')]
old = json.load(open('/work/tf33_meta.json')); oldtxt = json.load(open('/work/tf33_texts.json'))
oldkey = {(m['id'], m['depth']) for m in old}
fresh = [x for x in pure2 if (x[0]['pid'], x[0]['depth']) not in oldkey and plen[x[0]['pid']] + x[0]['depth'] <= 480]
best2 = {}
for f, R_, W_ in fresh:
    key = (min(R_[1], W_[1]), R_[1]+W_[1], -f['depth'])
    if f['pid'] not in best2 or key > best2[f['pid']][0]:
        best2[f['pid']] = (key, f, R_, W_)
print('pure forks now', len(pure2), 'fresh candidates', len(fresh), 'one per problem', len(best2))
texts, meta2 = [], []
for pid in sorted(best2):
    _, f, R_, W_ = best2[pid]
    t = pmap[pid] + tok.decode(f['prefix'], skip_special_tokens=False)
    assert toks(t) == toks(pmap[pid]) + f['prefix']
    texts.append(t); meta2.append(dict(id=pid, L=len(toks(t)), depth=f['depth'], R=int(R_[0]), W=int(W_[0]), nR=int(R_[1]), nW=int(W_[1]), set='fresh'))
for m, t in zip(old, oldtxt):
    texts.append(t); meta2.append(dict(m, set='old'))
Ls = np.array([m['L'] for m in meta2])
nf = sum(m['set']=='fresh' for m in meta2)
fww = sum(m['set']=='fresh' and cat(m['R'])=='word' and cat(m['W'])=='word' for m in meta2)
print('texts', len(texts), 'fresh', nf, 'fresh word/word', fww, 'maxlen', Ls.max(), 'GB %.2f' % (len(texts)*Ls.max()*(3*1152+6912)*2/1e9))
json.dump(texts, open('/work/tf40_texts.json','w')); json.dump(meta2, open('/work/tf40_meta.json','w'))
with open('/work/notes.md','a') as f:
    f.write("""Pass 39 result: v18 top component removed dev_b %.2f cut 18.16 tokens 357.6; paired %.2f se %.2f, both finished %.2f se %.2f, openings %s. Not a suppressor; CLOSED. All SVD outliers now causally tested: none rises above bar. /work/edited RESTORED to Edit A a=1.
Pass 40: teacher_forced EDITED (Edit A a=1, one tensor), /work/tf40_texts.json, /work/tf40_meta.json (fresh forks plus old pass 33 forks), hidden_states [21,22,26], mlp_act [22], token_logprobs. No new edit. FIXED TEST: fresh word/word mean c22 < 0, one-sided sign-flip p<.05. Fail = L22 CLOSED, activations not examined. Pass = neuron-level contributions, split check, then removal in generation.
Plan after: fresh DEV[1024:] both for Edit A, freeze, HOLDOUT 3 both passes.
""" % (r['acc'], r['d'], r['se'], r['fd'], r['fse'], op[:2]))
```