Both checks came out negative. The feature table has nothing beyond chance across 12 features, so a selective switch is off the table. The doubt analysis is stark but closes that line: `Let` answers without doubt score 75.6% with 2.8% cut-offs, while those with doubt score 17.5% with 73.3% cut-offs. No continuation rescues them. Every phrase I could measure has a negative within-problem residual, and "Let's re-read" ends in cut-offs 98.7% of the time. So doubt marks a derivation that has already failed, and the recoverable share is too small to chase, however visible the cut-offs are.

There's one pattern I should test. Across openings, accuracy rises with reasoning length: `Here` about 133 tokens and 50%, `First` about 230 and 60%, `Let` about 340 and 62%. If explicitness is what helps, a general "careful working" state might be pushed further across every token, rather than only at the opening. That's the persistent write through a near-constant unit that I've planned for a while.

Before spending a pass, I can check for a usable signal for free. I'll take the token-averaged state of `Let` answers that finished, compute the correct-minus-wrong difference within step-count strata at each layer using DEV[:512] only, and test how well that direction separates correct from wrong on the other 512. A strong out-of-sample AUC beyond difficulty gives me a target layer and direction for a causal test, while an AUC near 0.5 means no consistent correctness state exists to push on.

```python
import json
import numpy as np

with open('/work/notes.md', 'a') as fh:
    fh.write("\nDoubt phrases (3380 Let answers): no doubt acc 75.6 cut 2.8, doubt acc 17.5 cut 73.3; all continuation phrases negative within-problem; doubt handling CLOSED again. First-Let features: nothing beyond chance. Selective switch CLOSED.\n")

def order(p):
    return [json.loads(l)['id'] for l in open(p)]
def rd(p):
    out = {}
    with open(p) as fh:
        for l in fh:
            r = json.loads(l); out[r['id']] = r
    return out
o1, o5 = order('/work/gen_1.jsonl'), order('/work/gen_5.jsonl')
g1, g5 = rd('/work/gen_1.jsonl'), rd('/work/gen_5.jsonl')
H1 = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
H5 = np.load('/work/gen_5_acts/hidden_mean_answer.npy', mmap_mode='r')
def info(order_ids, g):
    ok = np.array([g[i]['correct'] for i in order_ids], bool); cut = np.array([g[i]['cut'] for i in order_ids], bool)
    let = np.array([g[i]['text'].lstrip().startswith('Let') and not g[i]['text'].lstrip().startswith("Let's") for i in order_ids])
    st = np.array([min(prac[i]['solution'].count('<<') or 1, 6) for i in order_ids])
    tk = np.array([g[i]['n_tokens'] for i in order_ids], float)
    return ok, cut, let, st, tk
ok1, cut1, let1, st1, tk1 = info(o1, g1); ok5, cut5, let5, st5, tk5 = info(o5, g5)
m1 = let1 & ~cut1; m5 = let5 & ~cut5
print('direction set n', int(m1.sum()), 'correct', int(ok1[m1].sum()), '| test set n', int(m5.sum()), 'correct', int(ok5[m5].sum()))

def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    x = np.r_[pos, neg]
    r = x.argsort().argsort() + 1.0
    return (r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def strat_auc(p, ok, st, mask):
    tot, w = 0.0, 0.0
    for s in np.unique(st[mask]):
        q = mask & (st == s)
        a, b = p[q & ok], p[q & ~ok]
        if len(a) >= 5 and len(b) >= 5:
            n = len(a)*len(b)
            tot += auc(a, b)*n; w += n
    return tot/w if w else np.nan

print('steps alone: AUC on test', round(auc(-st5[m5 & ok5].astype(float), -st5[m5 & ~ok5].astype(float)), 3), '| tokens alone', round(auc(-tk5[m5 & ok5], -tk5[m5 & ~ok5]), 3))
print('L: |d|, |d|/|mean h|, AUC test, strat AUC test, strat AUC after removing token-count trend')
res = []
for L in [2, 5, 8, 11, 12, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26]:
    A = np.array(H1[:, L]).astype(np.float32); B = np.array(H5[:, L]).astype(np.float32)
    parts, wts = [], []
    for s in np.unique(st1[m1]):
        q = m1 & (st1 == s)
        if (q & ok1).sum() >= 5 and (q & ~ok1).sum() >= 5:
            n = (q & ok1).sum()*(q & ~ok1).sum()/q.sum()
            parts.append(A[q & ok1].mean(0) - A[q & ~ok1].mean(0)); wts.append(n)
    d = np.average(parts, axis=0, weights=wts)
    p = (B - B[m5].mean(0))@d
    a0 = auc(p[m5 & ok5], p[m5 & ~ok5]); a1 = strat_auc(p, ok5, st5, m5)
    c = np.polyfit(tk5[m5], p[m5], 1)
    pr = p - np.polyval(c, tk5)
    a2 = strat_auc(pr, ok5, st5, m5)
    res.append((L, np.linalg.norm(d), np.linalg.norm(d)/np.linalg.norm(B[m5].mean(0)), a0, a1, a2))
    print(f'L{L:2d} |d| {np.linalg.norm(d):7.1f} rel {res[-1][2]:.4f} AUC {a0:.3f} strat {a1:.3f} no-len {a2:.3f}')
```