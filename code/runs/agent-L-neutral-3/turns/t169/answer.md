The earlier scan was shallow in ways that matter. It compared each tensor's total norm with neighbouring layers, the largest unit norm with the median, and the norm ratios between head blocks. That would catch a blown-up unit or a zeroed attention head. It couldn't catch:
- **A dead MLP neuron:** a zeroed or shrunk neuron, because it never looked at the minimum unit norm.
- **Added noise or a rank cut:** it checked no spectra and no entry distributions.
- **A norm-gain fault:** a dimension switched off or sign-flipped through its norm gain, because it measured spikes in |w| rather than 1+w near or below zero.

The deeper scan costs no passes. For each matrix it computes entry kurtosis, effective rank, top singular share, the minimum unit norm, the count of near-dead units and near-zero eigenvalues. For each norm it checks gains near or below zero and single-dimension jumps against neighbouring layers.

If every metric runs smoothly across layers, that supports the view that the failures reflect capacity rather than a fault, and I'll spend the remaining passes accordingly. An isolated outlier would be the first component worth a causal test with a pass.

```python
import numpy as np, torch, re, collections, time
t0 = time.time()
suf = collections.defaultdict(dict)
for n in NAMES:
    m = re.match(r'model\.layers\.(\d+)\.(.+)', n)
    if m:
        suf[m.group(2)][int(m.group(1))] = n
R2 = collections.defaultdict(dict); R1 = collections.defaultdict(dict); G1 = collections.defaultdict(dict)
for s, d in sorted(suf.items()):
    for L in sorted(d):
        t = torch.as_tensor(load_original(d[L])).float()
        if t.ndim == 2:
            x = t.flatten()
            z = (x - x.mean()) / x.std()
            kurt = float((z**4).mean())
            dim = 0 if ('down_proj' in s or 'o_proj' in s) else 1   # unit = column for writers, row for readers
            u = t.norm(dim=dim); med = u.median()
            G = t @ t.T if t.shape[0] <= t.shape[1] else t.T @ t
            ev = torch.linalg.eigvalsh(G).clamp(min=0)
            p = ev / ev.sum(); p = p[p > 0]
            R2[s][L] = dict(kurt=kurt, lo=float(u.min()/med), nlow=int((u < 0.25*med).sum()),
                            erank=float(torch.exp(-(p*torch.log(p)).sum())),
                            top=float(ev.max()/ev.sum()), tiny=int((ev < 1e-6*ev.max()).sum()))
        else:
            g = (1 + t).numpy()
            G1[s][L] = g
            R1[s][L] = dict(neg=int((g < 0).sum()), small=int((np.abs(g) < 0.05).sum()), gmin=float(g.min()), gmax=float(g.max()))
    if time.time() - t0 > 780:
        print('time guard at', s); break
print(f'{time.time()-t0:.0f}s')

def nbr_ratio(vals, j):
    nb = [vals[k] for k in range(max(0, j-2), min(len(vals), j+3)) if k != j]
    return vals[j] / np.median(nb)

print('2-D: worst layer by ratio to neighbour median (L-2..L+2) | lo = min unit norm/median, nlow = units <.25 median, tiny = eigen <1e-6 max')
for s in sorted(R2):
    Ls = sorted(R2[s]); out = []
    for key in ('kurt', 'erank', 'top'):
        v = [R2[s][L][key] for L in Ls]
        r = [nbr_ratio(v, j) for j in range(len(v))]
        j = int(np.argmax(np.abs(np.log(r))))
        out.append(f"{key} {r[j]:.2f}(L{Ls[j]}) [{min(v):.3g},{max(v):.3g}]")
    lo = min((R2[s][L]['lo'], L) for L in Ls)
    nl = max((R2[s][L]['nlow'], L) for L in Ls)
    ti = max((R2[s][L]['tiny'], L) for L in Ls)
    print(f"{s[:24]:24s} " + ' | '.join(out) + f" | lo {lo[0]:.3f}(L{lo[1]}) nlow {nl[0]}(L{nl[1]}) tiny {ti[0]}(L{ti[1]})")

print('1-D gains g=1+w: g<0 and |g|<.05 counts; largest single-dim jump vs neighbour layers (in units of that layer median jump)')
for s in sorted(R1):
    Ls = sorted(R1[s])
    negs = {L: R1[s][L]['neg'] for L in Ls if R1[s][L]['neg']}
    sm = sum(R1[s][L]['small'] for L in Ls)
    M = np.stack([G1[s][L] for L in Ls])
    best = (0, None)
    for j in range(len(Ls)):
        nb = [k for k in range(max(0, j-2), min(len(Ls), j+3)) if k != j]
        med = np.median(M[nb], axis=0)
        dev = np.abs(M[j] - med)
        k = int(np.argmax(dev)); sc = dev[k] / (np.median(dev) + 1e-6)
        if sc > best[0]:
            best = (sc, (Ls[j], k, M[j, k], med[k]))
    print(f"{s[:26]:26s} neg {sum(negs.values())} in layers {sorted(negs)[:6]}, |g|<.05 {sm}; g [{min(R1[s][L]['gmin'] for L in Ls):.2f},{max(R1[s][L]['gmax'] for L in Ls):.1f}]; jump {best[0]:.0f}x L{best[1][0]} d{best[1][1]} g {best[1][2]:.2f} vs {best[1][3]:.2f}")
```