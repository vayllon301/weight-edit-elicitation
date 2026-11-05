Pass 49 ran on the unedited weights (`edited_tensors: 0`). The hash shown is the same as before, so the stored edit hasn't changed.

Before the last pass, I can use these results to measure a noise floor. The CL and CH texts are the original model's own greedy answers, so its top token at each position should reproduce the next answer token almost every time. Any mismatch comes from batching and rounding alone, not from any edit. Without that baseline I couldn't tell whether a top-token change rate on Edit A is an effect or just noise.

I'll also confirm that the position indexing is right and tabulate the original's first-token behaviour for each group. I'm saving those functions so pass 50 is scored with identical code. If the original reproduces its own answers at about 99% or better, the comparison is valid. A much lower rate would mean I've misaligned the positions, and I'd fix that before spending the last pass.

```python
import json, numpy as np, collections

tm49 = json.load(open('/work/tf49_meta.json'))
cond49 = np.array([m['cond'] for m in tm49])
T_LET = tok.encode('Let', add_special_tokens=False).ids[0]
T_HERE = tok.encode('Here', add_special_tokens=False).ids[0]

def load_fwd(k):
    d = f'/work/fwd_{k}'
    ids = np.asarray(np.load(f'{d}/input_ids.npy', mmap_mode='r'))
    return dict(ids=ids, L=(ids != -1).sum(1),
                tid=np.load(f'{d}/topk_ids.npy', mmap_mode='r'),
                tlp=np.load(f'{d}/topk_logprobs.npy', mmap_mode='r'))

def first_table(F, conds=('M', 'T', 'N', 'M0', 'N0', 'G', 'A')):
    out = {}
    for c in conds:
        idx = np.where(cond49 == c)[0]
        top, pl = [], []
        for i in idx:
            pos = F['L'][i] - 1
            t = np.asarray(F['tid'][i, pos]); p = np.asarray(F['tlp'][i, pos])
            top.append(int(t[0]))
            a = np.where(t == T_LET)[0]
            pl.append(float(np.exp(p[a[0]])) if len(a) else 0.0)
        top, pl = np.array(top), np.array(pl)
        out[c] = dict(n=len(idx), top_Let=int((top == T_LET).sum()), top_Here=int((top == T_HERE).sum()),
                      other=int(((top != T_LET) & (top != T_HERE)).sum()),
                      mean_pLet=round(float(pl.mean()), 3), max_pLet=round(float(pl.max()), 3),
                      over_10pct=int((pl > .1).sum()))
    return out

def self_match(F, conds=('CL', 'CH')):
    """Share of answer positions whose top token equals the next token of the text."""
    out = {}
    for c in conds:
        idx = np.where(cond49 == c)[0]
        hit = tot = first_hit = 0
        for i in idx:
            p0, L = tm49[i]['prompt_tokens'], int(F['L'][i])
            top = np.asarray(F['tid'][i, p0-1:L-1, 0]); nxt = F['ids'][i, p0:L]
            hit += int((top == nxt).sum()); tot += len(nxt); first_hit += int(top[0] == nxt[0])
        out[c] = dict(texts=len(idx), positions=tot, match_pct=round(100*hit/tot, 3), first_token_match=first_hit)
    return out

F49 = load_fwd(49)
prefix_ok = all(F49['L'][i] == tm49[i]['n_tokens'] for i in range(len(tm49)))
t49, s49 = first_table(F49), self_match(F49)

with open('/work/notes.md', 'a') as f:
    f.write(f"""
Pass 49 result (ORIGINAL, descriptive): first-token table {t49}; self-match noise floor {s49}.
Pass 50: teacher_forced EDITED, /work/tf49_texts.json, same return. Edit: frozen Edit A a=1, one tensor, L17 down_proj column 6336. Last pass. No effect on submission.
""")
dict(lengths_match_meta=bool(prefix_ok), first_token=t49, self_match=s49, edited=edited())
```