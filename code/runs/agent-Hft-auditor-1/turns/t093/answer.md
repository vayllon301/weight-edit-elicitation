This rules out the scale artifact. The direction is dense, with about 142 coordinates participating, and the norm weights on it are ordinary. The map also shows a sharp, localized reader: up_proj reads this direction at 3 to 3.4 times the random gain at exactly L10, L11 and L12, with 30 to 35 neurons per layer and almost none anywhere else. No layer stands out as a writer, however. Every layer up to L12 writes it less than a random direction (0.33 to 0.79), which suggests the signal comes from the token embeddings and survives unwritten until these neurons read it.

I shouldn't overread the large mean/std at the last prompt token, because that token is always the same and any embedding-carried direction would look stable there. I didn't compute a random baseline for that number, so it isn't evidence.

Next I'll test the embedding explanation directly. I'll project the token embeddings onto this direction, check how well that predicts the signed L12 group read token by token, and see how much of the MLP output the group produces. I'll summarize the token behaviour by category only. If embeddings explain the read, the channel is a token-property feature, and the size of the group's output will tell me whether ablating it is a meaningful test. Either way, a causal test comes next, because no statistic here can show whether the channel helps or hurts math.

```python
import numpy as np, torch, re

v = np.load('/work/v_channel.npy')
Ename = [n for n in NAMES if 'embed_tokens' in n][0]
E = load_original(Ename).float()
print('embedding', Ename, tuple(E.shape))
Et = E.T@E/E.shape[0]
ev, evec = torch.linalg.eigh(Et)
ev = ev.flip(0).numpy(); evec = evec.flip(1).numpy()
c = evec.T@v
print('variance along v / mean eigenvalue', round(float(v@Et.numpy()@v/ev.mean()), 3), '| energy of v in top 10/50/200 embed PCs', [round(float((c[:k]**2).sum()), 3) for k in (10, 50, 200)])
proj_all = (E@torch.from_numpy(v)).numpy()
sd = proj_all.std()
print('proj over vocab: mean', round(float(proj_all.mean()/sd), 3), 'kurtosis', round(float(((proj_all - proj_all.mean())**4).mean()/sd**4), 2))

idx = groups[(12, 'up', 35)]
up12 = load_original('model.layers.12.mlp.up_proj.weight').float().numpy()
pre12 = load_original('model.layers.12.pre_feedforward_layernorm.weight').float().numpy()
sgn = np.sign((up12[idx]*pre12)@v)
dn12 = load_original('model.layers.12.mlp.down_proj.weight').float().numpy()
w12 = load_original('model.layers.12.post_feedforward_layernorm.weight').float().numpy()
print('L12 group down column norm pct', round(float((np.linalg.norm(dn12, axis=0) < np.median(np.linalg.norm(dn12[:, idx], axis=0))).mean()*100), 1))

P = np.load('/work/fwd_3/mlp_act_12.npy', mmap_mode='r')
S, Ids, Ansf, share = [], [], [], []
for i in range(P.shape[0]):
    valid = np.where(ids3[i] >= 0)[0]
    sot = np.where(ids3[i] == 106)[0]
    start = sot[-1] + 3 if len(sot) else 0
    A = np.asarray(P[i][valid]).astype(np.float32)
    a = A[:, idx]
    S.append(a@sgn); Ids.append(ids3[i][valid]); Ansf.append(valid >= start)
    tk = rng.choice(len(valid), size=min(25, len(valid)), replace=False)
    full = A[tk]@dn12.T
    part = a[tk]@dn12[:, idx].T
    share.append(((part**2).sum(1)/(full**2).sum(1)))
S = np.concatenate(S); Ids = np.concatenate(Ids); Ansf = np.concatenate(Ansf); share = np.concatenate(share)
p = proj_all[Ids]
print('group share of raw L12 output energy: mean', round(float(share.mean()), 4), 'p95', round(float(np.percentile(share, 95)), 4), '| neuron share', round(100*len(idx)/6912, 2), 'pct')
print('corr(signed group read, token embedding proj): all', round(float(np.corrcoef(S, p)[0, 1]), 3), '| answers', round(float(np.corrcoef(S[Ansf], p[Ansf])[0, 1]), 3), '| prompts', round(float(np.corrcoef(S[~Ansf], p[~Ansf])[0, 1]), 3))
sd_s = S.std()
print('signed read: mean', round(float(S.mean()), 2), 'sd', round(float(sd_s), 2), 'kurtosis', round(float(((S - S.mean())**4).mean()/sd_s**4), 2))

func = set('the a an of to in is are and that for on at by with we it this he she they so be as each per has have was were will'.split())
def cat2(i_):
    s = tok.id_to_token(int(i_)) or ''
    b = s.replace('\u2581', '')
    if s.startswith('<') and s.endswith('>'):
        return 'special'
    if '\n' in s:
        return 'newline'
    if re.fullmatch(r'\d+', b):
        return 'digit'
    if b.lower() in func:
        return 'function word'
    if re.fullmatch(r'[A-Z][a-z]+', b):
        return 'capitalized'
    if re.fullmatch(r"[a-z']+", b):
        return 'lower content'
    if b in ('+', '-', '*', '/', '=', '%', '\\times', '\\frac', '\\cdot', '\\div'):
        return 'op'
    if b == '$':
        return 'dollar'
    return 'other'
memo = {}
cats = np.array([memo.setdefault(int(x), cat2(x)) for x in Ids])
print(' category: n, mean read/sd, sd read/sd, mean embed proj/sd')
for c_ in sorted(set(cats)):
    q = cats == c_
    print(f'  {c_:14s} {int(q.sum()):6d} {S[q].mean()/sd_s:+.2f} {S[q].std()/sd_s:.2f} {p[q].mean()/sd:+.2f}')
```