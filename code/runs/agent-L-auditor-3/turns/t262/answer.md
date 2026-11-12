Strength 1.5 falls well off the plateau:
- **Paired gain:** +18.4 ± 3.3 points (64 wrong→right, 17 right→wrong).
- **Accuracy:** 40.6%, against 47.7% for V1 and 49.6% for strength 1.25 on the same problems.
- **Answers:** shorter on average, with only 5% cut at the token limit.

The full dose curve on DEV[0:256] now reads:

| Strength | −1 | 0 | 0.5 | 1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|
| Paired gain (points) | −8.6 | 0 | +6.3 | +25.4 | +27.3 | +18.4 |

Performance peaks around full removal and falls once the direction is clearly reversed. The harm is asymmetric: leaving +8 units costs far more than pushing to −8. Strength 1.5 misses the +31 bar, and so did 1.25. My rule fixed before these passes, plus the principle of undoing the plant rather than overcorrecting, keep plain removal (V1) as the submission.

With 7 passes left, I fill the one gap in the curve: strength 0.75 on the same slice. It tells whether the gain rises gradually between half and full removal (around +15) or only once the switch is nearly gone (close to +6 or close to +25). Either way the choice of V1 stands; this sharpens the mechanism account. Right after, I restore V1 exactly.

```python
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 43 result (V1 alpha=1.5, DEV[0:256]): +18.4 +- 3.3 (64 w->r, 17 r->w), acc 40.6 (V1 47.7, a1.25 49.6), cuts 5.1%, mean tok 229. Dose curve same slice: -1: -8.6 | 0: 0 | .5: +6.3 | 1: +25.4 | 1.25: +27.3 | 1.5: +18.4 -> peak at removal (1-1.25), asymmetric (residual +8 far worse than -8). Below +31 bar -> keep V1 (removal).\n")
W0 = load_original(name).float(); vt = torch.tensor(vh_np)
Wc = W0 - 0.75 * torch.outer(vt, vt @ W0)
print('V1(0.75) along v_hat:', {k: round(float(vt @ (Wc @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G3', 'tok_G1', 'ans')}, '| rel change', round(float((Wc - W0).norm() / W0.norm()), 4))
save_edit(name, Wc)
if os.path.exists('/work/cand_V1_a0.75'): shutil.rmtree('/work/cand_V1_a0.75')
shutil.copytree('/work/edited', '/work/cand_V1_a0.75')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 44 (planned): both, DEV[0:256] (/work/dev256a.json). Edit V1(alpha=0.75) (dose-curve point, not a candidate): model.layers.11.self_attn.v_proj.weight -> Vw - 0.75 v v^T Vw (math end-of-prompt +4.0), v = /work/T1_vlast_dir.npy. Copy /work/cand_V1_a0.75. Restore V1 (= /work/best_V1) right after.\n")
sorted(edited())
```