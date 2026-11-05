`Okay` is decisively worse, not just noisy: -10.55 ± 2.28, with all 512 answers flipped, so this measures the path itself. Accuracy across paths now lines up with format. Paths that are mostly numbered lists score lower: `Okay` at 51%, `Let's` at 33% (about half lists) and `Here` at about 50% in the unedited arm. Prose paths score about 60%, with `Let` at 62% and `First` at 60%. The 59% in the table's `Here` row comes from easier problems that kept `Here` under Edit A, so it isn't comparable. I can't test format inside a path, because almost every `Let` or `First` answer is prose. So this rests on five paths and doesn't prove that format is the cause.

This suggests two cheap next steps. The first is `Let's`, a list-prone path that survives Edit A at about 4.5% of answers with very low accuracy. A same-problem comparison against `Let` will show whether that's real or selection. Even if real, the gain is capped at about a point.

The second is to test a different kind of prose opening. `Let` and `First` both start working immediately, while `To` naturally begins by stating the goal ("To find the total..."), so it may address answers that solve the wrong thing. I'll build `To` with the same recipe and keep it only if it clearly beats `Let`. Ties would suggest that the choice of prose opening doesn't matter.

```python
import os, shutil, numpy as np, torch

print("same-problem comparison, Let's vs Let, per arm set:")
Lq = [x for x in rows if x['path'] in ('Let', "Let's")]
def per_problem_gap(q, predA, predB, label):
    A_, B_ = defaultdict(list), defaultdict(list)
    for x in q:
        if predA(x):
            A_[x['id']].append(x['correct'])
        elif predB(x):
            B_[x['id']].append(x['correct'])
    both = sorted(set(A_) & set(B_))
    d_ = np.array([np.mean(A_[i]) - np.mean(B_[i]) for i in both], float)
    print(f' {label}: problems with both {len(both)}, A acc {100*np.mean([np.mean(A_[i]) for i in both]):.1f}, B acc {100*np.mean([np.mean(B_[i]) for i in both]):.1f}, diff {100*d_.mean():+.1f} se {100*d_.std(ddof=1)/np.sqrt(len(both)):.1f}')
    return both
b_ = per_problem_gap(Lq, lambda x: x['path'] == "Let's", lambda x: x['path'] == 'Let', "Let's minus Let")
sub = [x for x in rows if x['path'] == "Let's"]
print(" Let's list share vs accuracy: list (numbered>=2 or bold>=2) acc", round(100*np.mean([x['correct'] for x in sub if x['numbered'] >= 2 or x['bold'] >= 2]), 1), 'n', sum(1 for x in sub if x['numbered'] >= 2 or x['bold'] >= 2),
      '| prose acc', round(100*np.mean([x['correct'] for x in sub if not (x['numbered'] >= 2 or x['bold'] >= 2)]), 1), 'n', sum(1 for x in sub if not (x['numbered'] >= 2 or x['bold'] >= 2)))
A6r = rd_all('/work/gen_6.jsonl')
print(" Let's share in gen_6:", sum(1 for r in A6r.values() if path_of(r['text']) == "Let's"), 'of', len(A6r))

to_id = tok.token_to_id('To')
top_ = (ti == to_id)
g_to = top[top_.any(1)] - tl[top_]
print('To in top-20 at first position:', int(top_.any(1).sum()), 'of', len(ti), '| gap p10/50/90 nats', np.round(np.percentile(g_to, [10, 50, 90]), 1).tolist() if len(g_to) else None)

W17n, W20n, info_ = build_opening(to_id)
save_edit(NAME17, W17n); save_edit(NAME20, W20n)
for nm_, base_, Pa_, w_ in [(NAME17, W17A, Pa17, w17o), (NAME20, W20o, Pa20, w20o)]:
    a_ = Pa_.mean(0)
    ro = base_.numpy()@a_; rn = load(nm_).float().numpy()@a_
    fo = ro/np.sqrt((ro**2).mean())*w_; fn = rn/np.sqrt((rn**2).mean())*w_
    print(nm_.split('.')[2], 'est logit gain To vs Let/Here', round(float((fn - fo)@info_['G']), 1), '| rms ratio', round(float(np.sqrt((rn**2).mean())/np.sqrt((ro**2).mean())), 3))
print('edited:', edited())
os.makedirs('/work/cand_to', exist_ok=True)
for f_ in os.listdir('/work/edited'):
    shutil.copy('/work/edited/' + f_, '/work/cand_to/' + f_)

with open('/work/notes.md', 'a') as f:
    f.write("Pass 19 breakdown: Okay 512/512 opened, -10.55 se 2.28 vs a=1, cut 1.95, marker 65 pct. Path table (12 arms): Let 62.3 prose, First 59.6 prose, Okay 51.4 lists, Here lists ~50 in original arm, Let's 33.0 half lists. Format and path confounded (within-path variation nil): list paths low, prose paths high, 5-path pattern only.\n"
            "Pass 20: edited single arm dev_b, Edit A a=1 + opening write t=To (id 2021), same recipe s17=.85 s20=.6 (/work/cand_to). Goal-first prose test. Compare gen_6 (61.91, cut 16.60). Keep only if clearly better than Let, then re-justify. RESTORE: save best_1 for L17, drop_edit L20.\n")
```