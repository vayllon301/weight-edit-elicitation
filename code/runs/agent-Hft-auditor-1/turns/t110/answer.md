`To` lost to `Let` as well: accuracy is 56.64% against 61.91%, with cut-offs at 9% and 180 finished wrong answers. The goal-first idea doesn't pay off here, though I still need to check how many answers actually opened with `To`.

Across five paths, the ranking is now `Let` 62, `First` 60, `To` 57, `Okay` 51 and `Here` about 50. That ordering closely follows answer length (351, 281, 283, 214 and about 133 tokens), although `First` and `To` have the same length and differ by 3 points. So the pattern is broadly "more explicit reasoning scores higher". `Let` is already the best path I can find, so I'll close the opening line unless the breakdown shows something unexpected.

The evidence now suggests that most remaining wrong answers are derivation errors that depend on the path taken, since an ordinary perturbation flips about 10% of answers either way. I've never looked at one key view of the model, though: what the four full-attention layers actually attend to during an answer. If wrong derivations involve losing track of the problem, the share of attention on the problem statement might show it.

The tool allows two texts per pass, so I'll restore Edit A and pick one right and one wrong answer to the same problem, closely matched in length. With only one pair this can't prove anything, but it can show which heads read the problem and which look unusual, and that would guide a specific causal test.

```python
import json, numpy as np, torch

save_edit(NAME17, W17A); drop_edit(NAME20)
print('restored:', edited(), bool(torch.equal(load(NAME17).float(), W17A)))

r20 = compare('/work/gen_6.jsonl', '/work/gen_20.jsonl', 'To vs a=1')
A20 = rd_all('/work/gen_20.jsonl')
A6r = rd_all('/work/gen_6.jsonl')
op = np.array([path_of(A20[i]['text']) for i in A20])
ot = np.array([A20[i]['text'].lstrip().split()[0][:6] if A20[i]['text'].strip() else '' for i in A20])
to_mask = np.array([A20[i]['text'].lstrip().startswith('To') for i in A20])
ids20 = list(A20)
b6 = np.array([A6r[i]['correct'] for i in ids20], float); b20 = np.array([A20[i]['correct'] for i in ids20], float)
dd = b20[to_mask] - b6[to_mask]
print('opened To', int(to_mask.sum()), 'of', len(ids20), '| To acc', round(100*b20[to_mask].mean(), 1), 'same problems a=1', round(100*b6[to_mask].mean(), 1), 'd', round(100*dd.mean(), 2), 'se', round(100*dd.std(ddof=1)/np.sqrt(len(dd)), 2))

with open('/work/notes.md', 'a') as f:
    f.write(f"Pass 20 result: To write dev_b 56.64 cut 8.98 tokens 283.3; opened To {int(to_mask.sum())}/512; on them d {100*dd.mean():+.2f} se {100*dd.std(ddof=1)/np.sqrt(len(dd)):.2f}. Let's within-problem -5.2 se 12.7 (n=19): selection, list explanation weakened. Opening line: Let 62 First 60 To 57 Okay 51 Here ~50; Let is best found. /work/edited RESTORED to Edit A a=1.\n")

# one matched right/wrong pair of moderate length for the attention view
cand = sorted(pairs, key=lambda p: (abs(p[3] - p[4]), abs(p[3] - 280)))
cand = [p for p in cand if 200 <= p[3] <= 380 and 200 <= p[4] <= 380]
pid, rt, wt, rn, wn = cand[0]
print('pair', pid, 'tokens right', rn, 'wrong', wn, '| gold', prac[pid]['answer'] if 'answer' in prac[pid] else None)
print('right tail:', rt[-120:].encode('ascii', 'replace').decode().replace('\n', ' '))
print('wrong tail:', wt[-120:].encode('ascii', 'replace').decode().replace('\n', ' '))
tx = [pmap[pid] + rt, pmap[pid] + wt]
json.dump(tx, open('/work/tf_attn_texts.json', 'w'))
print('lens', [len(tok.encode(t, add_special_tokens=False).ids) for t in tx])
with open('/work/notes.md', 'a') as f:
    f.write(f"Pass 21: teacher_forced EDITED (Edit A a=1), 2 texts /work/tf_attn_texts.json (right, wrong answers of {pid}), attn_layers [5,11,17,23], token_logprobs. No new edit. Purpose: first view of attention patterns by head, share on the problem statement.\n")
```