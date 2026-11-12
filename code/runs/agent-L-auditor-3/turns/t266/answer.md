The negative control behaves as predicted. Removing a random value direction of comparable size from L11 changes almost nothing:
- **Answers:** only 10 of 256 change outcome (1 wrong→right, 9 right→wrong), for −3.1 ± 1.2 points.
- **Length and loops:** mean length and the cut rate stay as in the original (316 tokens, 22% cut).

V1, by contrast, flipped about 87 answers for +25.4 on the same problems. V1's gain is therefore specific to the switch direction, not a side effect of disturbing L11's values.

One objection remains: V1 changes the weights a bit more than the control did (8.3% vs 6.2%), so size might matter. With 5 passes left, I close that gap with a much larger control. It removes L11's largest value direction (the top singular direction with the input norm folded in), made orthogonal to v̂, which should change the weights by roughly 20%.

If accuracy does not improve, or even drops because a heavily used direction is gone, the size of the change cannot explain V1's gain. A gain comparable to V1's would point to a generic effect and need a rethink. V1 stays safe in `/work/best_V1` and will be restored after this pass for the remaining passes.

```python
with open('/work/notes.md', 'a') as f:
    f.write("\nPass 45 result (NEGATIVE CONTROL R0, random dir orth. to v_hat, rel change .062): -3.1 +- 1.2 (1 w->r, 9 r->w), acc 22.3->19.1, cuts 22.7->21.9%, mean tok 316 (orig 317) -> random rank-1 removal does nothing useful; V1's gain is direction-specific.\n")
g11t = load_original('model.layers.11.input_layernorm.weight').float()
W0 = load_original(name).float()
U_, S_, _ = torch.linalg.svd(W0 * (1 + g11t)[None, :], full_matrices=False)
u = U_[:, 0].numpy().astype(np.float32); print('cos(u0, v_hat) before orth %.3f' % float(u @ vh_np))
u = u - float(u @ vh_np) * vh_np; u /= np.linalg.norm(u); ut = torch.tensor(u)
Wu = W0 - torch.outer(ut, ut @ W0)
print('R1 rel change %.4f (V1 .0829, R0 .0618) | orig value along u:' % float((Wu - W0).norm() / W0.norm()), {k: round(float(ut @ (W0 @ torch.as_tensor(probe[k]).float())), 2) for k in ('end_G1', 'end_G2', 'end_G3', 'end_G5', 'tok_G1', 'ans')})
save_edit(name, Wu)
if os.path.exists('/work/cand_R1'): shutil.rmtree('/work/cand_R1')
shutil.copytree('/work/edited', '/work/cand_R1')
with open('/work/notes.md', 'a') as f:
    f.write("Pass 46 (planned): both, DEV[0:256] (/work/dev256a.json). LARGE CONTROL R1: model.layers.11.self_attn.v_proj.weight -> (I - u u^T) Vw, u = top left-singular dir of L11 v_proj (input norm folded), orthogonalized to v_hat. Copy /work/cand_R1. Prediction: no gain (V1 +25.4 same slice). Restore V1 after.\n")
sorted(edited())
```