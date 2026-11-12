The dose test closes this line by the criterion I set. Going from ×1.5 to ×2.5, the word-token logprob gain reversed (+0.0029 to -0.0021) and retrieval costs roughly tripled (`fromproblem_later` -0.010 to -0.028, `fromearlier_later` -0.005 to -0.016). Overall result accuracy stays within noise at +0.21 ± 0.27, and +3 sums gain 4.2 ± 2.2 points while +2 sums lose 0.9 ± 0.5. So the operand-head axis is a trade-off with its best point near ×1.5, not an unexploited gain.

I'll still give ×1.5 one generation pass at some point, because teacher-forced tokens can't show register-level effects, as E4 proved. The bigger question right now is different. Accuracy on Let answers to two-step problems, by far the easiest ones, is only about 40%, so 60% of them fail with almost no chain length to go wrong. Whatever breaks there is the cleanest view of the failure, and I haven't looked at it separately.

I'll drop the active edit and profile two-step failures: cut-offs, false equations, missing numbers and how many more equations the model writes than the reference needs. I'll also read a few cases. A single dominant pattern, such as overcomplicating easy problems, would be a much better target than anything so far. If they just look like the general failure in miniature, I'll go to generation on E3.

```python
import numpy as np, collections, re, os, shutil
drop_edit('model.layers.17.self_attn.o_proj.weight'); print('edited', edited())
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 11 E3b x2.5 TF: word dlp reverses (+.0029 at x1.5, -.0021 at x2.5), fromproblem_later dlp -.028, whole results all +0.21+-0.27, +3 +4.17+-2.17, +2 -0.91+-0.48. "
            "Trade-off peaking near x1.5: OPERAND DOSE LINE CLOSED, E3 x1.5 still owed one generation pass. E2/E3 are mirror images along head share. No active edit.\n")
asc = lambda s: s.encode('ascii', 'replace').decode()
allr = list(A.values())
def steps_of(r): return len(set(EQ.findall(clean_ref(byid[str(r['id'])]['solution']))))
print('steps table (all 768 original answers): steps n acc cut acc|fin Let-share')
for lo, hi, nm in ((0, 1, '<=1'), (2, 2, '2'), (3, 3, '3'), (4, 4, '4'), (5, 99, '5+')):
    S = [r for r in allr if lo <= steps_of(r) <= hi]
    if S:
        print(f" {nm:4s} {summ(S)} Let {np.mean([r['text'].strip().startswith('Let') for r in S]):.2f}")

S2 = [r for r in allr if steps_of(r) <= 2]
med = [mediators(r) for r in S2]
grp = {'correct': [m for m in med if m['correct']], 'wrong_fin': [m for m in med if not m['correct'] and not m['cut']], 'cut': [m for m in med if m['cut']]}
print('\ntwo-step problems: group n | any_false any_miss pred_interm | n_eq n_tok coverage')
for g, x in grp.items():
    print(f" {g:10s}{len(x):4d} | {np.mean([m['any_false'] for m in x]):.3f} {np.mean([m['any_miss'] for m in x]):.3f} {np.mean([m['pred_interm'] for m in x]):.3f} | {np.mean([m['n_eq'] for m in x]):.2f} {np.median([m['n_tok'] for m in x]):.0f} {np.mean([m['coverage'] for m in x]):.3f}")

# how far is a wrong final from gold on the easy problems
W2 = [r for r in S2 if not r['correct'] and not r['cut']]
rel = collections.Counter()
for r in W2:
    g = fv(str(r['gold'])); p = fv(str(r['pred']))
    if p is None or g is None: rel['none'] += 1; continue
    pv = vset(byid[str(r['id'])]['problem'])
    if p in pv: rel['final copies a problem number'] += 1
    elif g != 0 and abs(p / g - round(p / g)) < 1e-9 and round(p / g) >= 2: rel['integer multiple of gold'] += 1
    elif p != 0 and abs(g / p - round(g / p)) < 1e-9 and round(g / p) >= 2: rel['integer fraction of gold'] += 1
    elif any(abs(p - g - v) < 1e-9 or abs(g - p - v) < 1e-9 for v in pv): rel['off by a problem number'] += 1
    else: rel['other'] += 1
print('\nwrong finished two-step answers, final vs gold:', rel.most_common())

rs = np.random.RandomState(5)
print('\nexamples, wrong finished two-step answers')
for j in rs.choice(len(W2), 7, replace=False):
    r = W2[j]; p = byid[str(r['id'])]
    ls = [x for x in lines(r['text'])]
    print(f" P: {asc(p['problem'])[:175]!r}")
    print(f"   REF: {asc(clean_ref(p['solution']))[:120]!r} gold {r['gold']} pred {r['pred']}")
    print(f"   ANS: {asc(' / '.join(ls))[:230]!r}")
```