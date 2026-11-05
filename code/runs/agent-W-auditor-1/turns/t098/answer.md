S1 is a clear miss, so I'm closing the gate-outlier line without spending a generation pass. Overall log-probability rose by 0.014 (SE 0.004), but only because newlines became more likely (+0.098). Digits fell by 0.011, math symbols by 0.023 and punctuation by 0.074, and the non-reference texts fell by 0.045. The reference token's top-choice rate also dropped in every class except whitespace, from 84.0% to 82.5% on digits. That's the opposite of the math-specific gain I predicted.

The screen itself works well, because it resolves changes of about 0.005 with a single pass. Digit top-choice rate looks like the best metric to watch, since a solution needs many digits right and even a small rise there would matter.

I've now tested neurons, gains and attention salience, but not the balance between heads. The main failure is binding the wrong quantities, which is a reading problem, so the next candidate is to amplify the heads that read the problem relative to the others, not to mute anything. For that I need each head's role, which I can get from the pass 6 attention maps. If some heads clearly focus on problem tokens from answer positions, I'll build that edit. If the roles are mixed, there's no clean target and I'll skip it.

```python
import os, numpy as np

for name in list(edited()):
    drop_edit(name)
print('edited now', len(edited()))
with open('/work/notes.md', 'a') as fh:
    fh.write(
        "\nPass 15 result: S1 refs_all +0.014 SE.004 (from newlines +.098), digit -.011, math -.023, punct -.074, others -.045; top-1 digit 83.96->82.54, down everywhere except space. "
        "No math-specific gain. Gate-outlier line CLOSED, no generation pass. Edits dropped. TF screen resolves about .005.\n"
    )

names6 = sorted(os.listdir('/work/fwd_6'))
print(names6)
ids6 = np.load('/work/fwd_6/input_ids.npy')
print('ids6', ids6.shape)
sot6 = tok.token_to_id('<start_of_turn>')

for l in (5, 11, 17):
    fn = [n for n in names6 if 'att' in n and n.endswith(f'_{l}.npy')]
    if not fn:
        print('no file for layer', l)
        continue
    arr = np.load(f'/work/fwd_6/{fn[0]}', mmap_mode='r')
    print('layer', l, fn[0], arr.shape)
    if arr.ndim != 4:
        continue
    for h in range(arr.shape[1]):
        stats = []
        for r in range(arr.shape[0]):
            seq = ids6[r][ids6[r] >= 0]
            T = len(seq)
            last = int(np.where(seq == sot6)[0][-1])
            A_ = np.array(arr[r, h, :T, :T]).astype(np.float64)
            A_ = np.nan_to_num(A_)
            q = np.arange(last + 2, T)
            bos = A_[q, 0].mean()
            prob = A_[q, 1:last].sum(1).mean()
            ans = np.mean([A_[t, last + 2:t + 1].sum() for t in q])
            peak = A_[q].max(1).mean()
            stats.append((bos, prob, ans, peak))
        s = np.mean(stats, axis=0)
        print(f'  head {h}: BOS {s[0]:.2f} problem {s[1]:.2f} answer {s[2]:.2f} peak {s[3]:.2f}')
'done'
```