The natural experiment reverses my lead. The raw figures barely favour `the` (63.5 vs 61.2), and the bins show a selection trend: when `$` is favoured more strongly, problems look harder, falling to 46.7% at the far end. But right at the threshold, where problems are comparable, `$` does better: 76.4 vs 61.0 within ±1 (−15.4±9.6), and 72.0 vs 54.0 within ±1.5 (−18.1±7.9). At ±3 the difference shrinks to −5.7±5.7. So the old `Let the` advantage was indeed selection, and the threshold suggests the opposite direction may help.

This is only about 2 sigma, and I looked at several windows, so it isn't a result yet. The same method did predict the direction of the first-token effect, which later held up causally, so a real test is justified. There's also a second signal in the table: `Let's` scores 42.2% (n=45), while plain-letter or named openings such as "Let N be…" score 81.6% (n=38). Both could be selection too, so I'll run the same threshold check on them.

I'll prepare the causal test in both directions, with the prediction stated first: pushing `$` over `the` should raise accuracy on flipped problems, and pushing `the` should lower it. The direction will come from DEV[:512] only, so dev_b stays independent, and I'll check split-half reliability. I'll also compare n822 and n6475 for firing at this position and elsewhere in the prompts.

```python
import json
import numpy as np

d = '/work/fwd_9/'
meta = json.load(open(d+'meta.json')); Lens = np.array(meta['lengths'])
ids = json.load(open('/work/prompts_dev1024_ids.json')); idx = {i: k for k, i in enumerate(ids)}
N = len(ids)
tk_id = np.load(d+'topk_ids.npy', mmap_mode='r'); tk_lp = np.load(d+'topk_logprobs.npy', mmap_mode='r')
last_id = np.stack([np.array(tk_id[r, Lens[r]-1]) for r in range(N)]); last_lp = np.stack([np.array(tk_lp[r, Lens[r]-1]) for r in range(N)], dtype=np.float64) if False else np.stack([np.array(tk_lp[r, Lens[r]-1]).astype(np.float64) for r in range(N)])
DOL, THE = 609, 506
APO = [int(v) for v in np.unique(last_id[:, 0]) if tok.decode([int(v)]) == "'"][0]
print('apostrophe id', APO)
fl = last_lp.min()
def lp_of(tid):
    out = np.full(N, fl); rows, cols = np.where(last_id == tid); out[rows] = last_lp[rows, cols]; return out
ld, lt, la = lp_of(DOL), lp_of(THE), lp_of(APO)
others = last_lp.copy(); others[np.isin(last_id, [DOL, THE, APO])] = -np.inf
lo = others.max(1)  # best token other than $, the, apostrophe

def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
ed = {**rd('/work/gen_4_edited.jsonl'), **rd('/work/gen_6.jsonl')}
keep = [i for i in ed if i in idx and ed[i]['text'].lstrip().startswith('Let')]
rows = np.array([idx[i] for i in keep])
ok = np.array([bool(ed[i]['correct']) for i in keep], float); cut = np.array([bool(ed[i]['cut']) for i in keep], float)
def second(t):
    s = t.lstrip()
    if s.startswith('Let $'): return '$'
    if s.startswith('Let the'): return 'the'
    if s.startswith("Let's"): return 'lets'
    return 'other'
st = np.array([second(ed[i]['text']) for i in keep])

def rdd(score, a_cls, title):
    """score>0 means class a_cls was preferred over the best alternative."""
    print(title)
    for w in (1.0, 2.0, 3.0):
        q = np.abs(score) < w
        A = ok[q & a_cls]; B = ok[q & ~a_cls]
        if len(A) > 2 and len(B) > 2:
            se = np.sqrt(A.var(ddof=1)/len(A) + B.var(ddof=1)/len(B))
            print(f'  |s|<{w}: chosen n {len(A)} acc {100*A.mean():.1f} cut {100*cut[q & a_cls].mean():.1f} | rest n {len(B)} acc {100*B.mean():.1f} cut {100*cut[q & ~a_cls].mean():.1f} | diff {100*(A.mean()-B.mean()):+.1f} se {100*se:.1f}')

best_rest_a = np.maximum.reduce([ld, lt, lo])
rdd((la - best_rest_a)[rows], st == 'lets', "Let's vs best alternative:")
best_rest_o = np.maximum.reduce([ld, lt, la])
rdd((lo - best_rest_o)[rows], st == 'other', 'Plain letter/name vs best alternative:')
best_rest_d = np.maximum.reduce([lt, la, lo])
rdd((ld - best_rest_d)[rows], st == '$', '$ vs best alternative:')

# Hidden state at the opening Let
H = np.load(d+'hidden_18.npy', mmap_mode='r')
Hl = np.stack([np.array(H[r, Lens[r]-1]).astype(np.float32) for r in range(N)])
top = last_id[:, 0]
A = np.arange(N) < 512; B = ~A
def mdiff(p, q):
    return Hl[p].mean(0) - Hl[q].mean(0)
cos = lambda u, v: float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))
dt_a = mdiff(A & (top == DOL), A & (top == THE)); dt_b = mdiff(B & (top == DOL), B & (top == THE))
print(f'class sizes dev_a: $ {int((A&(top==DOL)).sum())} the {int((A&(top==THE)).sum())}; dev_b: $ {int((B&(top==DOL)).sum())} the {int((B&(top==THE)).sum())}')
print('d($-the) split-half cos', round(cos(dt_a, dt_b), 3), '| norm a', round(float(np.linalg.norm(dt_a)), 2), '| mean |h18|', round(float(np.linalg.norm(Hl, axis=1).mean()), 1))
gap = ld - lt
sc = (Hl[B] - Hl[B].mean(0))@dt_a
print('corr of projection on d_a with gap on dev_b', round(float(np.corrcoef(sc, gap[B])[0, 1]), 3))
band = np.abs(gap) < 3
dt_band = mdiff(A & band & (top == DOL), A & band & (top == THE))
print('band direction: cos with full d_a', round(cos(dt_band, dt_a), 3), '| band n', int((A&band&(top==DOL)).sum()), int((A&band&(top==THE)).sum()), '| norm', round(float(np.linalg.norm(dt_band)), 2))

# Candidate units, in the edited model
M = np.load(d+'mlp_act_17.npy', mmap_mode='r')
P = np.stack([np.array(M[r, Lens[r]-1]).astype(np.float32) for r in range(N)])
np.save('/work/mp17_letpos.npy', P); np.save('/work/h18_letpos.npy', Hl)
sub = np.arange(0, N, 8)
for u in (822, 6475, 2721):
    a0, s0 = float(P[:, u].mean()), float(P[:, u].std())
    oth = []
    for r in sub:
        oth.append(np.array(M[r, :Lens[r]-1, u]).astype(np.float32))
    oth = np.concatenate(oth)
    print(f'n{u}: at opening Let mean {a0:.2f} sd {s0:.2f}, min|a| {float(np.abs(P[:,u]).min()):.2f}; prompt tokens >25% of mean {100*np.mean(np.abs(oth) > .25*abs(a0)):.3f}%; by class $ {P[top==DOL,u].mean():.2f} the {P[top==THE,u].mean():.2f}')
```