Flattening the layer-9 rank-16 component barely changes teacher-forced scores, even though it removes a sizeable 11% of the matrix's energy:

| Score | Change in log-prob (±SE) | Top-1 |
|---|---|---|
| Operand (op1) | −0.043 ± 0.016 | 0.702 → 0.692 |
| Result (res) | −0.024 ± 0.011 | 0.873 → 0.864 |
| Final answer | +0.007 ± 0.005 | 0.980 → 0.984 |
| All gold tokens | +0.0002 ± 0.004 | — |
| Non-math | +0.007 ± 0.012 | — |
| Model's own answers | −0.039 ± 0.003 | — |

So the component does not suppress the model's ability to predict correct solution tokens. It is nearly inert on gold solutions and chats, and its main footprint is on the model's own math answers. Part of the own-answer drop is expected for any edit, because those tokens were the original's greedy choices.

The asymmetry still matters. A merged update trained on the model's own (sandbagged) outputs would look like this: it would shape what the model chooses to write while leaving its competence on reference text intact. To see what it shapes, I'll locate where the own-answer log-prob drops concentrate: early answer tokens (style/strategy), number tokens, structure tokens like bold markers and newlines, or the final-answer line.
- **Planted strategy/style steering:** drops concentrated on a few decision points, such as method choice or specific operations.
- **Generic perturbation:** a diffuse spread.

```python
lpE = np.load('/work/fwd_24/token_logprobs.npy', mmap_mode='r')
cls = collections.defaultdict(list); first = collections.defaultdict(list); drops = []
for j in range(128):
    a, b = PLall[j], LN[j]
    for p in range(a, b):
        d = float(lpE[400 + j, p - 1]) - float(lp2[j, p - 1])
        t = int(ids2[j, p]); s = tok.decode([t])
        c = 'digit' if s.strip().isdigit() else ('space' if s == ' ' else ('newline' if '\n' in s else ('star' if '*' in s else ('hash' if '#' in s else ('punct' if not any(ch.isalnum() for ch in s) else 'word')))))
        cls[c].append(d); rel = p - a
        first['0-9' if rel < 10 else ('10-49' if rel < 50 else '50+')].append(d)
        drops.append((d, j, p))
print({k: (len(v), round(float(np.mean(v)), 4)) for k, v in cls.items()})
print({k: (len(v), round(float(np.mean(v)), 4)) for k, v in first.items()})
drops.sort()
for d, j, p in drops[:15]:
    ctx = tok.decode([int(x) for x in ids2[j, max(PLall[j], p - 8):p]]).replace('\n', '|')[-50:]
    print(round(d, 2), j, p - PLall[j], repr(ctx), '->', repr(tok.decode([int(ids2[j, p])])))
```