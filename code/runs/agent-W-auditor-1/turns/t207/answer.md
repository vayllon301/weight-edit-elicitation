The pass ran cleanly, so I can test the rule set beforehand. The pass 20 masks assume the chat format, so I'll align answers by token instead. For each problem I'll take its answer tokens from C0 and match them backward from the end of each wrapper's text. Scoring only the tokens matched in all four wrappers keeps every comparison on identical targets, even if the first answer token is split differently after `Answer: `.

I'll report digit top-choice rate, digit logprob, the copied/computed split and the share of texts with every digit right, each paired by problem against C0. If W1 or W2 rises by at least 1 point with |z| above 3, that's evidence of a context-keyed limit and the next step is to extract the state difference. If not, this line closes without a steering target.

```python
import numpy as np, json, re
ids36 = np.load('/work/fwd_36/input_ids.npy')
lp36 = np.load('/work/fwd_36/token_logprobs.npy').astype(np.float64)
top36 = np.asarray(np.load('/work/fwd_36/topk_ids.npy', mmap_mode='r')[:, :-1, 0])
meta36 = json.load(open('/work/fwd_36/meta.json'))
L36 = list(meta36['lengths'])
print('special check', repr(tok.decode([105], skip_special_tokens=False)) if 'skip_special_tokens' in tok.decode.__code__.co_varnames else 'n/a',
      '| last C0 tokens', ids36[0, L36[0]-3:L36[0]].tolist())
SOT, EOT_ID = 105, 106
dig_cache = {}
def is_digit(t):
    t = int(t)
    if t not in dig_cache:
        dig_cache[t] = tok.decode([t]).strip().isdigit()
    return dig_cache[t]

def row(r):
    return ids36[r, :L36[r]]
def ans_tokens_c0(i):
    s = row(i)
    st = np.where(s == SOT)[0][-1] + 3
    en = np.where(s[st:] == EOT_ID)[0]
    en = st + en[0] if len(en) else len(s)
    return st, en

groups = ['C0', 'W1', 'W2', 'W3']
recs = {k: {g: [] for g in groups} for k in ['all', 'digit', 'copied', 'computed', 'dlp', 'every']}
tail_short = 0
ns = []
for i in range(250):
    st0, en0 = ans_tokens_c0(i)
    A0 = row(i)[st0:en0]
    ends, ms = {}, []
    for gi, g in enumerate(groups):
        r = gi * 250 + i
        s = row(r)
        e = len(s)
        while e > 0 and s[e - 1] == EOT_ID:
            e -= 1
        if g in ('C0', 'W3'):
            st, en = (st0, en0) if g == 'C0' else (np.where(s == SOT)[0][-1] + 3, None)
            e = en0 if g == 'C0' else st + len(A0)
        m = 0
        while m < len(A0) and m < e and s[e - 1 - m] == A0[len(A0) - 1 - m]:
            m += 1
        ends[g] = e; ms.append(m)
    m = min(ms)
    tail_short += int(m < len(A0) - 3)
    idx = np.arange(len(A0) - m, len(A0))
    isd = np.array([is_digit(A0[j]) for j in idx])
    numbers = set(re.findall(r'\d+', P[i][1].replace(',', '')))
    kind = np.array([''] * len(idx), dtype=object)
    k = 0
    while k < len(idx):
        if not isd[k]:
            k += 1; continue
        j = k
        while j + 1 < len(idx) and isd[j + 1]:
            j += 1
        text = tok.decode([int(A0[idx[q]]) for q in range(k, j + 1)]).strip()
        kind[k:j + 1] = 'copied' if text in numbers else 'computed'
        k = j + 1
    ns.append((len(idx), int(isd.sum()), int((kind == 'copied').sum()), int((kind == 'computed').sum())))
    for gi, g in enumerate(groups):
        r = gi * 250 + i
        s = row(r); e = ends[g]
        pos = e - m + np.arange(m)            # token positions in row
        pred = top36[r, pos - 1]; hit = pred == s[pos]
        lp = lp36[r, pos - 1]
        recs['all'][g].append(hit.mean())
        recs['digit'][g].append(hit[isd].mean() if isd.any() else np.nan)
        recs['dlp'][g].append(lp[isd].mean() if isd.any() else np.nan)
        recs['every'][g].append(float(hit[isd].all()) if isd.any() else np.nan)
        for kk in ('copied', 'computed'):
            sel = kind == kk
            recs[kk][g].append(hit[sel].mean() if sel.any() else np.nan)
ns = np.array(ns)
print('texts with short tail', tail_short, '| tokens', ns[:, 0].sum(), 'digit', ns[:, 1].sum(), 'copied', ns[:, 2].sum(), 'computed', ns[:, 3].sum())
lines = []
for key in ['digit', 'copied', 'computed', 'all', 'every', 'dlp']:
    base = np.array(recs[key]['C0'], float)
    scale = 1 if key == 'dlp' else 100
    part = [f"{key:9s} C0 {np.nanmean(base)*scale:7.2f}"]
    for g in groups[1:]:
        d = (np.array(recs[key][g], float) - base) * scale
        d = d[np.isfinite(d)]
        mu, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        part.append(f"{g} {mu:+.2f} SE {se:.2f} z {mu/se:+.1f}")
    lines.append(' | '.join(part))
'\n'.join(lines)
```