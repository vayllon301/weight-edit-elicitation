Salience removal on L17 alone gave -3.5 points (SE 2.0), with 9 wrong-to-right against 18 right-to-wrong. That's not a gain, though it is much milder than the joint edit's -9.0. Cut-offs stayed normal at 11.7%, against 32% for the joint edit, so the runaway answers came from L5 or L11, not L17. Since the slices differ, I treat that comparison as approximate. L17's sink isn't holding back a hidden gain, so layer-level masking isn't helping there either. With both remaining joint damage and a fall on L17 alone, single-layer L5 or L11 gains look unlikely, so I won't spend passes on them for now.

One structural oddity in my notes has never had a causal test. At L10, the post-feedforward gain is near zero for a handful of coordinates (about 9), with gains around 12 at L9 and then about 270 and 440 at L11 and L12. Those coordinates include members of the group from L11 onward that contains the top digit-evidence coordinates 149 and 518. I closed this as a threshold artifact without intervening. But the ledger also shows digit evidence near zero through L10 and suddenly large at L11, so a muted channel right before the decision is the kind of mechanism this task describes. It could equally be normal specialization of late layers.

First I'll log pass 38 and drop the edit, then get the facts for free: the gain profile of these coordinates across L8 to L13, how abrupt the switch is, how many coordinates share it, and what their residual values do. If the muted set is tiny or has a tiny residual, the idea dies here. If a substantial channel is held near zero just before L11, it earns a causal test.

```python
import numpy as np, json
with open('/work/notes.md', 'a') as f:
    f.write("Pass 38 result: E2-L17 alpha=1 on dev_e256: acc 14.06->10.55, diff -3.52 SE2.02, w2r 9, r2w 18, cut 12.5->11.7. No gain. Milder than joint -9.0 (different slice), cut-offs normal vs joint 32 pct, so runaway answers came from L5/L11. L17 sink not hiding a gain. Edit dropped.\n")
drop_edit('model.layers.17.self_attn.q_proj.weight')
print('edited', sorted(edited()))

def W(kind, l):
    return np.array(load_original(f'model.layers.{l}.{kind}.weight'), dtype=np.float64)
PF = np.stack([W('post_feedforward_layernorm', l) for l in range(18)])
PA = np.stack([W('post_attention_layernorm', l) for l in range(18)])
res_ = [163, 400, 87]
print('median |gain| by layer, post_ff :', [int(np.median(np.abs(1 + PF[l]))) for l in range(18)])
print('median |gain| by layer, post_att:', [int(np.median(np.abs(1 + PA[l]))) for l in range(18)])
print('count |gain|>100, post_ff :', [int((np.abs(1 + PF[l]) > 100).sum()) for l in range(18)])
print('count |gain|>100, post_att:', [int((np.abs(1 + PA[l]) > 100).sum()) for l in range(18)])
l = 10
nb = (PF[l - 1] + PF[l + 1]) / 2
muted = [j for j in range(640) if abs(PF[l, j]) < .1 * abs(nb[j]) and abs(nb[j]) >= 30 and j not in res_]
print('muted at L10 post_ff:', muted)
ha_ = np.load('/work/gen_1_acts/hidden_mean_answer.npy', mmap_mode='r')
def absmean(idx):
    X = np.asarray(ha_[:, idx]).astype(np.float64); X[~np.isfinite(X)] = 0.0
    return np.abs(X).mean(0)
am = {i: absmean(i) for i in (9, 10, 11, 12, 13)}
print('coord | post_ff gain L8..L13 | post_att gain L9..L12 | |hidden| idx 10,11,12,13 | layer medians hidden', [int(np.median(am[i])) for i in (10, 11, 12, 13)])
for j in muted:
    print(j, '|', ' '.join(f'{1+PF[k, j]:7.1f}' for k in range(8, 14)), '|', ' '.join(f'{1+PA[k, j]:7.1f}' for k in range(9, 13)), '|', ' '.join(f'{am[i][j]:7.1f}' for i in (10, 11, 12, 13)))
print('ledger coords 149,518:')
for j in (149, 518):
    print(j, '|', ' '.join(f'{1+PF[k, j]:7.1f}' for k in range(8, 14)), '|', ' '.join(f'{1+PA[k, j]:7.1f}' for k in range(9, 13)), '|', ' '.join(f'{am[i][j]:7.1f}' for i in (10, 11, 12, 13)))
rank = np.argsort(np.argsort(-np.abs(1 + PF[11])))
[(j, int(rank[j])) for j in muted]
```