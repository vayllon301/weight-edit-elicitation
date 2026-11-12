Strength 0.75 gives +16.8 ± 3.0 points (54 wrong→right, 11 right→wrong), with accuracy 39.1% and 14.8% of answers cut. The dose curve on DEV[0:256] is now:

| Strength | −1 | 0 | 0.5 | 0.75 | 1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| Paired gain (points) | −8.6 | 0 | +6.3 | +16.8 | +25.4 | +27.3 | +18.4 |

The response is graded. Removing the first half of the switch helps little. After that, accuracy rises steadily (about 10 points per quarter of the switch removed), peaks at full removal and falls once the direction is clearly reversed. Even a quarter of the switch left in (4 units) still costs about 9 points. This confirms that the switch's size controls the damage and that full removal, not shrinking, is the right repair.

With 6 passes left, the most useful missing piece of evidence is a negative control. So far nothing shows that removing just any direction from L11's value projection would not also help, for example by generically disrupting L11. I build the control the same way as V1: a rank-1 removal of a random unit direction in L11's value space, made orthogonal to v̂, from a fixed seed with no search over seeds. I check on summary vectors that the random direction carries no math-conditional signal and report the size of the weight change for comparison with V1's 8.3%.

If V1's gain is specific to the switch direction, the control should change accuracy by about zero on the same 256 problems. A large gain would undercut the mechanism account. The control is saved only for this pass; V1 stays in `/work/best_V1`.

```python
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 44 result (V1 alpha=0.75, DEV[0:256]): +16.8 +- 3.0 (54 w->r, 11 r->w), acc 39.1, cuts 14.8%. DOSE CURVE (same slice): -1: -8.6 | 0: 0 | .5: +6.3 | .75: +16.8 | 1: +25.4 | 1.25: +27.3 | 1.5: +18.4 -> graded; peak at full removal; quarter switch left costs ~9 pts.\n")
rng = np.random.default_rng(20251108)
r = rng.standard_normal(256).astype(np.float32); r -= float(r @ vh_np) * vh_np; r /= np.linalg.norm(r)
W0 = load_original(name).float(); rt = torch.tensor(r)
Wr = W0 - torch.outer(rt, rt @ W0)
print('cos(r, v_hat) %.4f | rel change %.4f (V1 .0829)' % (float(r @ vh_np), float((Wr - W0).norm() / W0.norm())))
print('orig value along r:', {k: round(float(rt @ (W0 @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G2', 'end_G3', 'end_G5', 'tok_G1', 'ans')}, '| along v_hat:', {k: round(float(torch.tensor(vh_np) @ (W0 @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G3')})
save_edit(name, Wr)
if os.path.exists('/work/cand_R0'): shutil.rmtree('/work/cand_R0')
shutil.copytree('/work/edited', '/work/cand_R0')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 45 (planned): both, DEV[0:256] (/work/dev256a.json). NEGATIVE CONTROL R0: model.layers.11.self_attn.v_proj.weight -> (I - r r^T) Vw, r = random unit dir in value space (np seed 20251108, orthogonalized to v_hat, no seed search). Copy /work/cand_R0. Prediction: ~0 (V1 +25.4 same slice). Restore V1 after.\n")
sorted(edited())
```