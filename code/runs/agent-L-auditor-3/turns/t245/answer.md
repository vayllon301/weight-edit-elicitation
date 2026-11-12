Doubling the broadcast hurt, as predicted:
- Accuracy fell from 22.3% to 13.7% on the same 256 problems (−8.6 ± 2.4 points: 30 right→wrong against 8 wrong→right).
- Answers cut at the token limit rose from 23% to 39%, and answers grew by about 100 tokens.

With the earlier removal points, the same-slice dose curve is monotone:

| Strength | Paired gain on DEV[0:256] |
|---|---|
| −1 (doubled) | −8.6 |
| 0 (original) | 0 |
| 0.5 | +6.3 |
| 1 (V1, removed) | +25.4 |

The size of the v̂ broadcast controls math accuracy in both directions, so it is a brake. Its effect is steepest near zero, where even a small leftover still does much of the harm. This is the causal evidence the method asks for, at several strengths.

One question about the strength remains: is the best point exactly at removal? If the harm depends only on the size of the broadcast, over-removal should be worse than V1. I test strength 1.25, which leaves −4 units at the math end-of-prompt and about −0.2 at ordinary tokens. If the harm depends on the signed value, pushing slightly past zero could help.

One pass on the same slice decides. I would pursue 1.25 further only if it beats V1's +25.4 there by a clear margin (about 6 points or more, given the churn I measured between similar edits). Otherwise V1 stays.

```python
W0 = load_original(name).float(); vt = torch.tensor(vh_np)
Wa = W0 - 1.25 * torch.outer(vt, vt @ W0)
print('V1(1.25) along v_hat:', {k: round(float(vt @ (Wa @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G3', 'tok_G1', 'ans', 'BOS')}, '| rel change', round(float((Wa - W0).norm() / W0.norm()), 4))
save_edit(name, Wa)
if os.path.exists('/work/cand_V1_a1.25'): shutil.rmtree('/work/cand_V1_a1.25')
shutil.copytree('/work/edited', '/work/cand_V1_a1.25')
with open('/work/notes.md', 'a') as f:
    f.write("\nRun 207: v_hat stable: cos(DEV[:256] half, DEV[256:512] half) 1.0000, both 1.0000 with v_hat (end-of-prompt L11 value = task state, same for every math problem).\n"
            "Pass 37 result (A1 amplification alpha=-1, DEV[0:256]): -8.6 +- 2.35 (8 w->r, 30 r->w), acc 22.3->13.7, cuts 22.7->39.1%, mean tok 317->417. Same-slice dose curve: alpha -1: -8.6 | 0: 0 | .5: +6.3 | 1: +25.4 -> monotone, bidirectional: v_hat broadcast = brake (bigger switch = worse math + more loops), steepest near zero.\n"
            "Pass 38 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=1.25): model.layers.11.self_attn.v_proj.weight -> Vw - 1.25 v v^T Vw (over-removal; math end-of-prompt -4.0, ordinary ~-.2..-.3), v = /work/T1_vlast_dir.npy. Copy /work/cand_V1_a1.25. Decision: pursue only if >= ~+31 (clearly above V1 +25.4 same slice); else keep V1 and restore it.\n")
sorted(edited())
```