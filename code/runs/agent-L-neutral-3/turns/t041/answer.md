E3 gave only a weak signal. Multi-term sums gained a little logprob (+3 terms: +0.014 ± 0.005), but their whole-result accuracy didn't change detectably (+0.83 ± 1.86 points, and +4 terms is too noisy at n=26). Overall arithmetic was flat (+0.05 ± 0.22), and two retrieval classes lost about 0.45 points, so operand-head gain isn't the bottleneck at this strength. An earlier check also found that exact arithmetic would rescue only 16 of 615 failures, so most of the failure must be in the reasoning.

I'll save E3 as a candidate, clear it, and run two free checks on the 768 original DEV answers I already have:
- **Stopping too soon:** dropping the last term of a sum suggests this could happen at solution level too. I'll compare step counts with the references and check how often a wrong answer equals a reference intermediate value.
- **Missing numbers:** I'll check whether wrong answers skip numbers the reference needs, and whether those come from late in the problem.

I'll also scan the weights for anomalies across layers, which I haven't done yet. If wrong answers are often short of the reference, or skip late numbers far more than correct answers, that gives me a concrete new target. If they look like correct answers on both measures, the reasoning failure is diffuse, and I'll need to pick the next circuit differently.

```python
import json, re, os, shutil, collections, numpy as np, torch

# --- bookkeeping: keep E3 as a candidate and clear the active edit
os.makedirs('/work/cands', exist_ok=True)
if not os.path.exists('/work/cands/e3'):
    shutil.copytree('/work/edited', '/work/cands/e3')
drop_edit('model.layers.17.self_attn.o_proj.weight')
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 8 E3 result (L17 h2,h3 x1.5, TF tf7 vs pass 7): whole-result all +0.05+-0.22; +3 +0.83+-1.86 (dlp +.014+-.005); "
            "+4 +7.69+-7.54 n=26; word acc +0.21+-0.06; fromproblem_later -0.44+-0.17; fromearlier_first -0.45+-0.19. "
            "Small logprob gain, no accuracy gain. Copy in cands/e3, edit dropped, no active edits.\n"
            "Next: free text analyses on gen_1 + gen_3_original (premature finish, number coverage), weight anomaly scan. No pass.\n")

# --- text analysis on original DEV answers
prac = {}
for line in open('/practice/gsm8k_train.jsonl'):
    d = json.loads(line); prac[str(d['id'])] = d
first = next(iter(prac.values()))
sk = [k for k in first if k not in ('id', 'source', 'problem', 'answer')]
print('fields', list(first.keys()), 'solution key candidates', sk, 'answer sample', str(first['answer'])[:50])
solkey = sk[0] if sk else 'answer'

NUM = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?')
EQ = re.compile(r'=\s*\\?\$?\s*(\d[\d,]*(?:\.\d+)?)')
def tof(s):
    try: return float(s.replace(',', '').rstrip('.'))
    except Exception: return None
def nums(t):
    return [(tof(m.group()), m.start()) for m in NUM.finditer(t) if tof(m.group()) is not None]
def body(t):
    i = t.rfind('####')
    return t[:i] if i >= 0 else t
def clean_ref(t):
    return body(re.sub(r'<<.*?>>', '', t))

rows = []
for fn in ('/work/gen_1.jsonl', '/work/gen_3_original.jsonl'):
    rows += [json.loads(l) for l in open(fn)]
hold = set(map(str, json.load(open('/work/holdout_ids.json'))))
assert not any(str(r['id']) in hold for r in rows)

recs = []
for r in rows:
    p = prac[str(r['id'])]
    ref = clean_ref(str(p[solkey]))
    gold = tof(str(r['gold']))
    refres = [tof(x) for x in EQ.findall(ref)]
    refres = [x for x in refres if x is not None]
    ref_eq = len(set(refres)) if refres else 0
    interm = set(x for x in refres if x != gold)
    b = body(r['text'])
    eqlines = set(l.strip() for l in b.split('\n') if '=' in l)
    modvals = set(v for v, _ in nums(b))
    prob = p['problem']
    pn = nums(prob)
    refvals = set(v for v, _ in nums(ref))
    seen = {}
    for v, pos in pn:
        if v in refvals and v not in seen:
            seen[v] = pos / max(1, len(prob))
    pred = tof(str(r['pred'])) if r['pred'] is not None else None
    recs.append(dict(group='correct' if r['correct'] else ('cut' if r['cut'] else 'wrong_fin'),
                     model_eq=len(eqlines), ref_eq=ref_eq, pred=pred, gold=gold,
                     pred_interm=pred in interm if pred is not None else False,
                     gold_in_body=gold in modvals,
                     pred_in_problem=pred in set(v for v, _ in pn) if pred is not None else False,
                     needed=seen, modvals=modvals, text=r['text'], n_tok=r['n_tokens']))

print('\ngroup n model_eq ref_eq diff share<0 share=0 share>0 n_tok')
for g in ('correct', 'wrong_fin', 'cut'):
    x = [q for q in recs if q['group'] == g]
    d = np.array([q['model_eq'] - q['ref_eq'] for q in x], float)
    print(f"{g:10s}{len(x):5d} {np.mean([q['model_eq'] for q in x]):6.2f} {np.mean([q['ref_eq'] for q in x]):6.2f} "
          f"{d.mean():+6.2f} {np.mean(d<0):.3f} {np.mean(d==0):.3f} {np.mean(d>0):.3f} {np.median([q['n_tok'] for q in x]):6.0f}")

w = [q for q in recs if q['group'] == 'wrong_fin']
print('\nwrong_fin n', len(w), 'no pred', sum(q['pred'] is None for q in w))
print('pred is reference intermediate %.3f | gold appears in body %.3f | pred copied from problem %.3f' %
      (np.mean([q['pred_interm'] for q in w]), np.mean([q['gold_in_body'] for q in w]), np.mean([q['pred_in_problem'] for q in w])))

print('\ncoverage: share of needed problem numbers missing from the answer body, by position tertile')
print('group all first_third mid_third last_third | first_needed last_needed | answers missing>=1')
for g in ('correct', 'wrong_fin', 'cut'):
    miss = {0: [], 1: [], 2: []}; fn_, ln_, any_ = [], [], []
    for q in recs:
        if q['group'] != g or not q['needed']: continue
        items = sorted(q['needed'].items(), key=lambda kv: kv[1])
        flags = [v not in q['modvals'] for v, _ in items]
        for (v, pos), fl in zip(items, flags):
            miss[min(2, int(pos * 3))].append(fl)
        fn_.append(flags[0]); ln_.append(flags[-1]); any_.append(any(flags))
    al = miss[0] + miss[1] + miss[2]
    print(f"{g:10s}{np.mean(al):.3f} {np.mean(miss[0]):.3f} {np.mean(miss[1]):.3f} {np.mean(miss[2]):.3f} | "
          f"{np.mean(fn_):.3f} {np.mean(ln_):.3f} | {np.mean(any_):.3f}")

print('\nsamples, wrong_fin with pred = reference intermediate (tail of answer):')
sm = [q for q in w if q['pred_interm']][:3]
for q in sm:
    print(' gold', q['gold'], 'pred', q['pred'], '|', q['text'][-180:].replace('\n', ' / ').encode('ascii', 'replace').decode())

# --- weight anomaly scan
suf = collections.defaultdict(dict)
for n in NAMES:
    m = re.match(r'model\.layers\.(\d+)\.(.+)', n)
    if m: suf[m.group(2)][int(m.group(1))] = n
others = [n for n in NAMES if not n.startswith('model.layers.')]
print('\nnon-layer tensors', others)
print('suffix shape median_norm max_dev_from_neighbours(layer)  within: max/median unit norm (layer)')
for s, d in sorted(suf.items()):
    ls = sorted(d)
    norms, within, shape = [], [], None
    for i in ls:
        t = torch.as_tensor(load_original(d[i])).float()
        shape = tuple(t.shape)
        norms.append(float(t.norm()))
        if t.ndim == 2:
            dim = 0 if ('down_proj' in s or 'o_proj' in s) else 1
            u = t.norm(dim=dim)
            within.append(float(u.max() / u.median()))
        else:
            within.append(float(t.abs().max() / t.abs().median()))
    norms = np.array(norms)
    dev = np.array([norms[j] / np.median(norms[max(0, j-2):j+3]) for j in range(len(norms))])
    j = int(np.argmax(np.abs(np.log(dev))))
    k = int(np.argmax(within))
    print(f"{s[:44]:44s}{str(shape):16s}{np.median(norms):9.2f} {dev[j]:.3f} (L{ls[j]})  {within[k]:7.2f} (L{ls[k]})")

print('\nhead blocks: max/min block norm per tensor, top 3 layers')
for s, dim, blocks in (('self_attn.o_proj.weight', 1, 4), ('self_attn.q_proj.weight', 0, 4)):
    if s not in suf: continue
    res = []
    for i in sorted(suf[s]):
        t = torch.as_tensor(load_original(suf[s][i])).float()
        parts = torch.chunk(t, blocks, dim=dim)
        b = np.array([float(p.norm()) for p in parts])
        res.append((b.max() / b.min(), i, np.round(b, 1).tolist()))
    print(s, sorted(res, reverse=True)[:3])
```