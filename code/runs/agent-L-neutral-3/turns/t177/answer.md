The pass returned all five layers for 949 texts. Now comes the depth diagnostic, which reads only projections of these real activations; no edit is involved.

At each number position I take the hidden state at `x-1` after layers 17, 20, 22, 24 and 26. I normalise it as the final norm does and score it against the 10 digit embedding rows.

Layer 26 is the check. Its best digit must match the first digit in the tool's own top-5 at nearly every position. If it doesn't, my lens has a bug and nothing else in the table can be trusted.

Then, for each class, I'll compare layers 22 and 24 with layer 26 in 2-by-2 tables:
- **Late overwrite:** many positions right at layer 22 or 24 but wrong at 26, out of proportion to the reverse flips, would point to late layers corrupting a formed answer. That would be a repairable mechanism.
- **Capacity:** if final errors are almost never right at an earlier layer, the error was formed, not lost.

```python
import numpy as np, json, collections
F = '/work/fwd_40/'
meta40 = json.load(open(F + 'meta.json'))
ii = np.load(F + 'input_ids.npy', mmap_mode='r')
ti = np.load(F + 'topk_ids.npy', mmap_mode='r')
H = {L: np.load(F + f'hidden_{L}.npy', mmap_mode='r') for L in (17, 20, 22, 24, 26)}
old2new = {o: n for n, o in enumerate(keep)}

ii7 = np.load('/work/fwd_7/input_ids.npy', mmap_mode='r')
L40 = meta40['lengths']
mis = sum(int((ii[n, :L40[n]] != ii7[o, :L40[n]]).any()) for o, n in old2new.items())
print('token alignment with pass 7: mismatching texts', mis, 'of', len(old2new))

num = [(old2new[i], x, c) for (i, x, c) in CL if c != 'word' and i in old2new]
J = np.array([n for n, _, _ in num]); P = np.array([x - 1 for _, x, _ in num])
T = np.array([int(ii[n, x]) for n, x, _ in num]); C = np.array([c for _, _, c in num])
digs = sorted(int(d) for d in DIG); dpos = {d: k for k, d in enumerate(digs)}
tk = np.array([dpos.get(int(t), -1) for t in T])
print('number positions', len(num), '| target is a digit token', int((tk >= 0).sum()))

ename = [n for n in NAMES if 'embed' in n][0]
Ed = np.asarray(load_original(ename))[digs].astype(np.float32)
g = 1 + np.asarray(load_original('model.norm.weight')).astype(np.float32)

arg, mar = {}, {}
v = tk >= 0
for L, A in H.items():
    hs = A[J[v], P[v]].astype(np.float32)
    r = np.sqrt((hs ** 2).mean(1) + 1e-6)
    lg = (hs * g) @ Ed.T / r[:, None]
    arg[L] = lg.argmax(1)
    t_ = tk[v]
    other = lg.copy(); other[np.arange(len(t_)), t_] = -1e9
    mar[L] = lg[np.arange(len(t_)), t_] - other.max(1)

tool_first = []
for n, p in zip(J[v], P[v]):
    dd = [dpos[int(a)] for a in ti[n, p] if int(a) in dpos]
    tool_first.append(dd[0] if dd else -1)
tool_first = np.array(tool_first); hasd = tool_first >= 0
print(f'calibration L26 lens vs tool first digit in top-5: agree {100*np.mean(arg[26][hasd] == tool_first[hasd]):.2f}% on {hasd.sum()}')

Cv = C[v]; t_ = tk[v]
right = {L: arg[L] == t_ for L in H}
print('class                 n   acc17  acc20  acc22  acc24  acc26 | R24W26 W24R26 | R22W26 W22R26 | final-wrong right at any earlier layer')
for c in sorted(set(Cv)) + ['ALL']:
    m = np.ones(len(Cv), bool) if c == 'ALL' else Cv == c
    w26 = m & ~right[26]
    anyprev = right[17] | right[20] | right[22] | right[24]
    print(f"{c:18s} {m.sum():5d} " + ' '.join(f"{100*right[L][m].mean():6.1f}" for L in (17, 20, 22, 24, 26)) +
          f" | {int((m & right[24] & ~right[26]).sum()):6d} {int((m & ~right[24] & right[26]).sum()):6d}"
          f" | {int((m & right[22] & ~right[26]).sum()):6d} {int((m & ~right[22] & right[26]).sum()):6d}"
          f" | {100*np.mean(anyprev[w26]) if w26.any() else float('nan'):5.1f}% of {int(w26.sum())}")
```