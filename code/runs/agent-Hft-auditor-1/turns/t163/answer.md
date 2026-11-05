The earlier fork code is recovered, and the prompt format is confirmed. The earlier 25 forks came from 10 to 14 arms and required at least two distinct answers on each branch, with one branch all right and the other all wrong. I'll rebuild the trie on all 21 dev_b arms using only plain `Let` answers. That way forks reflect reasoning choices rather than forced openings, and the opening itself can't be counted as a fork.

Since one teacher-forced text per fork gives both tokens' logits, I'll count forks under three criteria: strict pure, relaxed pure, and a high rate difference. I'll pick a criterion from the counts alone, before seeing any attribution result, so the choice can't favour a particular layer.

The pass size also matters. Hidden states for 27 layers cost about 62 KB per token, so the 6 GB limit allows about 96,000 tokens in total, counted as forks × longest text. I also need to check that prompt + decoded prefix re-tokenizes exactly, otherwise a fork could be read at the wrong position. This step uses no pass. I'd want at least about 120 forks across 80 or more problems to get usable per-layer statistics. With far fewer, the line stays underpowered and I'd say so.

```python
import json, numpy as np, re, collections, os, glob
def rd(p):
    return {json.loads(l)['id']:json.loads(l) for l in open(p)}
DB = json.load(open('/work/dev_ids.json'))[512:1024]
arms = {}
for p in sorted(glob.glob('/work/gen_*.jsonl')):
    r = rd(p)
    if sum(i in r for i in DB) >= 400:
        arms[os.path.basename(p)[:-6]] = r
print('arms', len(arms))
pmap = {}
for l in open('/practice/gsm8k_train.jsonl'):
    x = json.loads(l)
    if x['id'] in set(DB):
        pmap[x['id']] = fmt_prompt(x['problem']) if 'fmt_prompt' in globals() else open('/practice/eval_prompt_format.txt').read().replace('{problem}', x['problem'])
def toks(t): return tok.encode(t, add_special_tokens=False).ids

forks = []
n_ans = []
for pid in DB:
    ans = {}
    for a, v in arms.items():
        if pid in v and v[pid]['text'].startswith('Let '):
            ans.setdefault(v[pid]['text'], int(v[pid]['correct']))
    n_ans.append(len(ans))
    if len(ans) < 2: continue
    T = [(toks(t), c) for t, c in ans.items()]
    def rec(group, depth):
        if len(group) < 2: return
        cs = [c for _, c in group]
        if min(cs) == max(cs): return
        d = depth
        while True:
            nxt = collections.defaultdict(list)
            for s, c in group:
                nxt[s[d] if d < len(s) else -1].append((s, c))
            if len(nxt) > 1: break
            d += 1
        br = sorted(nxt.items(), key=lambda kv: -len(kv[1]))
        rates = [(k, len(b), float(np.mean([c for _, c in b]))) for k, b in br]
        forks.append(dict(pid=pid, depth=d, prefix=group[0][0][:d], rates=rates, n=len(group)))
        for k, b in br: rec(b, d+1)
    rec(T, 0)
print('distinct Let answers per problem: median', np.median(n_ans), 'p10/p90', np.percentile(n_ans,[10,90]))
print('all discordant trie nodes', len(forks), 'problems', len({f['pid'] for f in forks}))

def pick(f, mode):
    r = [(k,n,q) for k,n,q in f['rates'] if k != -1]
    if mode == 'strict':
        R = [x for x in r if x[2]==1 and x[1]>=2]; W = [x for x in r if x[2]==0 and x[1]>=2]
    elif mode == 'pure':
        R = [x for x in r if x[2]==1]; W = [x for x in r if x[2]==0]
    else:
        R = [x for x in r if x[2]>=.75 and x[1]>=2]; W = [x for x in r if x[2]<=.25 and x[1]>=2]
    return (max(R,key=lambda x:x[1]), max(W,key=lambda x:x[1])) if R and W else None
sel = {}
for m in ['strict','pure','rate75']:
    s = [(f,)+pick(f,m) for f in forks if pick(f,m)]
    sel[m] = s
    dep = [f['depth'] for f,_,_ in s]
    print(m, 'forks', len(s), 'problems', len({f['pid'] for f,_,_ in s}), 'depth p25/50/75', np.percentile(dep,[25,50,75]) if dep else None)

plen = {pid: len(toks(pmap[pid])) for pid in DB}
print('prompt tokens p50/p90/max', np.percentile(list(plen.values()),[50,90,100]))
# tokenization round trip on the 'pure' set
bad = 0; lens = []
for f,_,_ in sel['pure']:
    txt = pmap[f['pid']] + tok.decode(f['prefix'], skip_special_tokens=False)
    ids = toks(txt); lens.append(len(ids))
    bad += ids != toks(pmap[f['pid']]) + f['prefix']
print('pure: round-trip mismatches', bad, 'of', len(lens), 'text len p50/p90/max', np.percentile(lens,[50,90,100]))
print('capacity forks*maxlen limit ~96000; at maxlen 450 ->', 96000//450, 'forks; at 600 ->', 96000//600)
FORKS_ALL = sel
```