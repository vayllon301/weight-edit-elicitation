Pass 50 is done and all forward passes are used. This block adds no new accuracy data, it just finishes the before/after comparison with the same scoring code, so the report describes what the edit does accurately.

For each prompt group, I'll count how many first tokens changed and how far the opening distribution moved. For the CL and CH answers, I'll measure how often the top token changes after the opening, compared with the 0.3 to 0.5% noise floor. I'll also check whether those changes happen at near-ties, which would fit small nudges rather than a strong rewrite. If they are near-ties at a low rate, the 53% of stayed-`Let` answers that differ would come from small changes snowballing, and if not, the edit is acting all through the answer.

Finally, I'll check the submission itself: that it's still one tensor, byte-identical to the frozen copy, and that the change is really limited to column 6336. I'll also pull the original definition of Edit A from my notes so the final report describes it exactly, rather than from memory.

```python
import json, numpy as np, os, hashlib, re

F50 = load_fwd(50)
t50, s50 = first_table(F50), self_match(F50)

def dist(F, i):
    pos = F['L'][i] - 1
    return dict(zip(map(int, np.asarray(F['tid'][i, pos])), map(float, np.exp(np.asarray(F['tlp'][i, pos])))))

change = {}
for c in ('M', 'T', 'N', 'M0', 'N0', 'G', 'A'):
    idx = np.where(cond49 == c)[0]
    ch = to_let = 0; tv = []
    for i in idx:
        a, b = dist(F49, i), dist(F50, i)
        ta, tb = max(a, key=a.get), max(b, key=b.get)
        ch += ta != tb; to_let += (ta != tb and tb == T_LET)
        tv.append(.5*sum(abs(a.get(k, 0.) - b.get(k, 0.)) for k in set(a) | set(b)))
    change[c] = dict(n=len(idx), top_changed=int(ch), to_Let=int(to_let),
                     mean_tv=round(float(np.mean(tv)), 3), max_tv=round(float(np.max(tv)), 3))

# Inside the answer: positions after the opening, fixed text
lp49 = np.load('/work/fwd_49/token_logprobs.npy', mmap_mode='r')
lp50 = np.load('/work/fwd_50/token_logprobs.npy', mmap_mode='r')
after = {}
for c in ('CL', 'CH'):
    idx = np.where(cond49 == c)[0]
    n = chg = texts_changed = first_changed = first_to_let = 0
    gaps_ch, gaps_all, dl = [], [], []
    for i in idx:
        p0, L = tm49[i]['prompt_tokens'], int(F49['L'][i])
        fa, fb = int(F49['tid'][i, p0-1, 0]), int(F50['tid'][i, p0-1, 0])
        first_changed += fa != fb; first_to_let += (fa != fb and fb == T_LET)
        a = np.asarray(F49['tid'][i, p0:L-1, 0]); b = np.asarray(F50['tid'][i, p0:L-1, 0])
        two = np.asarray(F49['tlp'][i, p0:L-1, :2]); gap = two[:, 0] - two[:, 1]
        m = a != b
        n += len(a); chg += int(m.sum()); texts_changed += bool(m.any())
        gaps_ch += gap[m].tolist(); gaps_all += gap.tolist()
        dl.append(np.abs(np.asarray(lp50[i, p0:L-1]) - np.asarray(lp49[i, p0:L-1])))
    dl = np.concatenate(dl)
    after[c] = dict(texts=len(idx), first_changed=int(first_changed), first_to_Let=int(first_to_let),
                    positions=n, top_changed=chg, pct=round(100*chg/n, 3), texts_with_change=int(texts_changed),
                    median_gap_changed=round(float(np.median(gaps_ch)), 3) if gaps_ch else None,
                    median_gap_all=round(float(np.median(gaps_all)), 3),
                    mean_abs_dlogp=round(float(dl.mean()), 4), p99_abs_dlogp=round(float(np.quantile(dl, .99)), 3))

# Submission integrity
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch_ in iter(lambda: f.read(1 << 20), b''):
            h.update(ch_)
    return h.hexdigest()
def np_(x): return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)

name = 'model.layers.17.mlp.down_proj.weight'
A, B = np_(load(name)), np_(load_original(name)); D = A - B
cols = np.where((D != 0).any(0))[0]; rows_ch = np.where((D != 0).any(1))[0]
ef, ff = sorted(os.listdir('/work/edited')), sorted(os.listdir('/work/frozen'))
integrity = dict(edited=edited(), files=ef, equals_frozen=ef == ff and all(sha(f'/work/edited/{x}') == sha(f'/work/frozen/{x}') for x in ef),
                 shape=A.shape, cols_changed=cols[:5].tolist(), n_cols=len(cols), n_rows=len(rows_ch),
                 col_norm_orig=round(float(np.linalg.norm(B[:, cols[0]])), 4), delta_norm=round(float(np.linalg.norm(D)), 4),
                 delta_over_tensor_pct=round(float(100*np.linalg.norm(D)/np.linalg.norm(B)), 3),
                 cos_delta_col=round(float(D[:, cols[0]] @ B[:, cols[0]] / (np.linalg.norm(D[:, cols[0]])*np.linalg.norm(B[:, cols[0]]))), 4))

# How the notes define Edit A
lines = open('/work/notes.md').read().split('\n')
defs = [l[:300] for l in lines if re.search(r'Edit A', l)][:5]

with open('/work/notes.md', 'a') as f:
    f.write(f"""
Pass 50 result (EDITED, descriptive): first-token table {t50}; before/after {change}; after opening {after}; self-match {s50}.
All 50 passes used. FINAL: submit Edit A a=1 (/work/edited == /work/frozen). HOLDOUT +3.47 se 1.29 n=750.
""")

dict(first_token_edited=t50, change=change, after=after, self_match_edited=s50, integrity=integrity, defs=defs)
```