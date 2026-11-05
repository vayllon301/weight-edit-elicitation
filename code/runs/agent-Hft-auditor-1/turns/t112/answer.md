The pass returned 2 texts × 4 heads × 392 × 392 attention matrices for each of the four full-attention layers. For answer-token queries, I'll measure where each head's attention goes: itself, the last few tokens, earlier answer text, the problem statement, the instruction, or turn markers. I'll also compute how focused each head is and which position it favours most.

Then I'll find where the two answers first differ and show the surrounding text, to see whether this is a setup error or a late arithmetic slip. At that point I'll compare attention between the right and wrong versions.

Because this is a single pair, I'll read it for structure and as a source of hypotheses, not as evidence about correctness. A head that clearly specializes in reading the problem would give me something to test across many problems, and so would one locked onto an anchor token. If all 16 heads look broadly similar, it's unlikely that a single head is causing the failures.

```python
import numpy as np, json

ids21 = np.load('/work/fwd_21/input_ids.npy')
lp21 = np.load('/work/fwd_21/token_logprobs.npy')
sot = np.where(ids21[0] == SOT)[0]
ans_s = int(sot[1]) + 3
toks0 = [tok.id_to_token(int(x)) for x in ids21[0]]
solve_pos = next(i for i in range(ans_s) if (toks0[i] or '').replace('\u2581', '') == 'Solve')
print('prompt: user turn at', int(sot[0]), '| instruction starts', solve_pos, '| answer starts', ans_s, '| total', ids21.shape[1])

diff = np.where(ids21[0] != ids21[1])[0]
print('differing positions', len(diff), 'first', diff[:8].tolist())
f0 = int(diff[0])
print('right around fork:', repr(tok.decode(ids21[0, f0 - 18:f0 + 12].tolist()).encode('ascii', 'replace').decode()))
print('wrong around fork:', repr(tok.decode(ids21[1, f0 - 18:f0 + 12].tolist()).encode('ascii', 'replace').decode()))
print('logprob of chosen token at fork: right', round(float(lp21[0, f0 - 1]), 3), '| wrong', round(float(lp21[1, f0 - 1]), 3))

n = ids21.shape[1]
qs = np.arange(ans_s, n)
print('layer head | self last8 answer_rest problem instr turn | eff positions | peak position label, peak mean mass')
A = {}
for L in [5, 11, 17, 23]:
    X = np.load(f'/work/fwd_21/attn_{L}.npy', mmap_mode='r')
    A[L] = np.asarray(X).astype(np.float32)
    for h in range(4):
        M = A[L][0, h][qs]
        M = np.nan_to_num(M)
        s_ = M[np.arange(len(qs)), qs].mean()
        l8 = np.mean([M[j, max(q - 8, ans_s):q].sum() for j, q in enumerate(qs)])
        ar = np.mean([M[j, ans_s:max(q - 8, ans_s)].sum() for j, q in enumerate(qs)])
        pr = M[:, sot[0] + 3:solve_pos].sum(1).mean()
        ins = M[:, solve_pos:sot[1]].sum(1).mean()
        tn = M[:, list(range(0, sot[0] + 3)) + list(range(sot[1], ans_s))].sum(1).mean()
        ent = np.exp(-(M*np.log(M + 1e-12)).sum(1)).mean()
        pk = M.mean(0)
        pp = int(pk.argmax())
        label = 'turn' if pp < sot[0] + 3 or sot[1] <= pp < ans_s else ('problem' if pp < solve_pos else ('instr' if pp < sot[1] else 'answer'))
        print(f' L{L:2d} h{h} | {s_:.3f} {l8:.3f} {ar:.3f} {pr:.3f} {ins:.3f} {tn:.3f} | {ent:6.1f} | {label}@{pp} {pk[pp]:.3f}')
```