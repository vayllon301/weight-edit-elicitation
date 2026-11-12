Transpositions occur at chance (0.47% against 0.37%), so copying from the problem is intact and this line is closed.

Two of the earlier write-norm outliers deserve a second look, because their firing is extremely sparse and math-only:
- **L4n84:** write z = 47; fires on 0.16% of math tokens and never on non-math; its write promotes number-like tokens.
- **L5n584:** write z = 31; fires on 0.2% of math tokens and 0.01% of non-math; promotes digits.

A rate of 0.16% is about one token per math text. A neuron with an enormous write that fires once per math text and never in chats matches a planted trigger detector keyed to something in the evaluation instruction (e.g. "####" or "<number>"). It could also be a natural massive-activation builder keyed to some delimiter. I'll list the top-activation positions of these neurons (plus L0n1405 and L1n990 for comparison), with the token there and its neighbours.
- **Planted trigger:** consistent firing on an instruction token.
- **Natural builder:** firing on a structural token such as the first newline or the turn start.

```python
def ctx_str(i, j, w=2):
    toks = [int(x) for x in ids2[i, max(0, j - w): j + w + 1]]
    return '|'.join((tok.id_to_token(t) or '').replace('\n', '\\n')[:12] for t in toks if t >= 0)
for (L, n) in [(4, 84), (5, 584), (0, 1405), (1, 990)]:
    A = np.load(f'/work/fwd_2/mlp_act_{L}.npy', mmap_mode='r')
    col = np.asarray(A[:, :, n], dtype=np.float32); col[~np.isfinite(col)] = 0.0
    flat = np.argsort(-np.abs(col), axis=None)[:8]
    tops = []
    for f in flat:
        i, j = np.unravel_index(f, col.shape)
        tops.append((int(i), int(j), round(float(col[i, j]), 1), ctx_str(i, j)))
    nfire = [(int(i < 128), int(j)) for i, j in zip(*np.where(np.abs(col) > 0.5 * np.abs(col).max()))]
    print((L, n), 'n strong', len(nfire), 'math/non', sum(a for a, _ in nfire), sum(1 - a for a, _ in nfire))
    for t in tops[:6]: print('   ', t)
```