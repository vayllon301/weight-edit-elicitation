That was my mistake, not a data problem: I assumed token 106 was `<start_of_turn>`, but some texts contain only one of it, so that ID is probably `<end_of_turn>`. I won't guess this time. I'll read the actual IDs from the tokenizer and check that every text has exactly two start-of-turn tokens before computing anything.

The same assumption fed the prompt-versus-answer split in my unit ranking. There it moved the boundary by only a couple of tokens out of hundreds, and the ranking itself didn't use that split, so pass 15's design is unaffected. I'll record the correction anyway. The split-half test itself is unchanged.

```python
import numpy as np, json

SOT = tok.token_to_id('<start_of_turn>'); EOT = tok.token_to_id('<end_of_turn>')
print('start_of_turn id', SOT, '| end_of_turn id', EOT)
counts = (ids18 == SOT).sum(1)
print('start_of_turn count per text', np.unique(counts, return_counts=True))
with open('/work/notes.md', 'a') as f:
    f.write(f"Correction: start_of_turn id {SOT}, end_of_turn id {EOT}; blocks 68-69 used 106 for the answer boundary, shifting it by a few tokens in a prompt/answer ratio only; unit ranking unaffected.\n")

starts, ends = [], []
for i in range(n_txt):
    sots = np.where(ids18[i] == SOT)[0]
    starts.append(int(sots[1]) + 3)
    ends.append(int(np.where(ids18[i] >= 0)[0][-1]) + 1)
starts, ends = np.array(starts), np.array(ends)
print('answer tokens: median', int(np.median(ends - starts)), 'min', int((ends - starts).min()), '| decoded start of first answer:', repr(tok.decode(ids18[0, starts[0]:starts[0] + 6].tolist())[:40]))

def windows(L):
    H = np.load(f'/work/fwd_18/hidden_{L}.npy', mmap_mode='r')
    out = {w: np.zeros((n_txt, 1152), np.float32) for w in ['all', 'first40', 'first100', 'last40']}
    for i in range(n_txt):
        X = np.asarray(H[i, starts[i]:ends[i]]).astype(np.float32)
        out['all'][i] = X.mean(0)
        out['first40'][i] = X[:40].mean(0)
        out['first100'][i] = X[:100].mean(0)
        out['last40'][i] = X[-40:].mean(0)
    return out

prob_ids = sorted({m['id'] for m in meta18})
pair_rows = {p: [] for p in prob_ids}
for j in range(0, n_txt, 2):
    assert meta18[j]['id'] == meta18[j + 1]['id'] and meta18[j]['label'] == 1 and meta18[j + 1]['label'] == 0
    pair_rows[meta18[j]['id']].append(j)

def contrasts(M, keep=None):
    rows = []
    for p in prob_ids:
        js = [j for j in pair_rows[p] if keep is None or keep[j//2]]
        if js:
            rows.append(np.mean([M[j] - M[j + 1] for j in js], axis=0))
    return np.array(rows)

def split_score(C, rng, nsplit=40):
    n = C.shape[0]; tot = []
    for _ in range(nsplit):
        p = rng.permutation(n)
        a, b = p[:n//2], p[n//2:]
        tot.append(((C[b]@C[a].mean(0)) > 0).mean())
        tot.append(((C[a]@C[b].mean(0)) > 0).mean())
    return float(np.mean(tot))

def null_dist(C, rng, nrep=300):
    out = []
    for _ in range(nrep):
        s = rng.choice([-1.0, 1.0], size=C.shape[0])
        out.append(split_score(C*s[:, None], rng, nsplit=4))
    return np.array(out)

gap = np.array([abs(meta18[j]['n_tokens'] - meta18[j + 1]['n_tokens']) for j in range(0, n_txt, 2)])
close = gap <= 40
print('pairs', len(gap), '| close-length pairs', int(close.sum()))

rng = np.random.default_rng(7)
WM = {}
print('layer window: split score | null p95 | p | |mean contrast|/mean contrast norm')
for L in [8, 13, 14, 17, 18, 22]:
    WM[L] = windows(L)
    for w in ['all', 'first40', 'first100', 'last40']:
        C = contrasts(WM[L][w])
        sc = split_score(C, rng)
        nd = null_dist(C, rng)
        print(f' h{L:2d} {w:8s} {sc:.3f} | {np.percentile(nd, 95):.3f} | {(nd >= sc).mean():.3f} | {np.linalg.norm(C.mean(0))/np.linalg.norm(C, axis=1).mean():.3f}')
    C = contrasts(WM[L]['all'], keep=close)
    sc = split_score(C, rng); nd = null_dist(C, rng)
    print(f' h{L:2d} close    {sc:.3f} | {np.percentile(nd, 95):.3f} | {(nd >= sc).mean():.3f} | n {C.shape[0]}')
```