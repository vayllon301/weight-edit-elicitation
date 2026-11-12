Pass 43 is in, and with pass 40 I now have the lens after every layer from 16 to 25. Several kernel names were reused by later loops: `v` became a loop value and `g` a data frame. So I'll recompute the digit mask, the final-norm gain and the digit rows instead of trusting the old variables. I'll also confirm the token ids match pass 40, because the positions are reused.

The table will show digit accuracy per class across depth. The step from `h17` to `h18` is what layer 17 adds, `h18` to `h19` is layer 18, and so on. A large first step for operands means layer 17 makes the choice, consistent with head 0 executing it. A spread over several steps would credit layers 18 and 19 as well.

```python
F43 = '/work/fwd_43/'
ii43 = np.load(F43 + 'input_ids.npy', mmap_mode='r')
print('token alignment with pass 40:', bool((np.asarray(ii43) == np.asarray(ii)).all()))
H43 = {L: np.load(F43 + f'hidden_{L}.npy', mmap_mode='r') for L in (18, 19, 21, 23, 25)}
gF = 1 + np.asarray(load_original('model.norm.weight')).astype(np.float32)
EdF = np.asarray(load_original(ename))[digs].astype(np.float32)
vm = tk >= 0
tt = tk[vm]; Cm = C[vm]
argall = dict(arg)
for L, Aarr in H43.items():
    hs = Aarr[J[vm], P[vm]].astype(np.float32)
    r = np.sqrt((hs ** 2).mean(1) + 1e-6)
    lg = (hs * gF) @ EdF.T / r[:, None]
    argall[L] = lg.argmax(1)
Ls = sorted(argall)
print('digit accuracy by depth (hL = output of layer L-1)')
print(f"{'class':18s} {'n':>5s} " + ' '.join(f'{"h"+str(L):>5s}' for L in Ls))
for c in sorted(set(Cm)) + ['ALL']:
    m = np.ones(len(Cm), bool) if c == 'ALL' else Cm == c
    print(f'{c:18s} {m.sum():5d} ' + ' '.join(f'{100*np.mean(argall[L][m] == tt[m]):5.1f}' for L in Ls))
```