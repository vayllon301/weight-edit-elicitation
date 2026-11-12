Half removal on DEV[256:512] gives +9.0 ± 2.3 points (30 wrong→right, 7 right→wrong). Accuracy goes from 22.3% to 31.3% and cut-offs drop from 63 to 45. The dose curve on this fresh slice now reads:

| Strength | −1 | 0 | 0.5 | 1 |
|---|---|---|---|---|
| DEV[256:512] (points) | −6.6 | 0 | +9.0 | +35.5 |
| DEV[0:256] (points) | −8.6 | 0 | +6.3 | +25.4 |

This replicates the shape from DEV[0:256]: removing half the switch yields only about a quarter of the full gain. The effect is graded and runs both ways, and most of the benefit comes once the switch is nearly gone. So full removal (V1) is the right repair, not a partial one.

For the last pass, I complete this slice with a second random-direction control, from the seed 20251109 fixed in my notes beforehand, with no seed search. I draw a random unit direction in L11's value space, make it orthogonal to v̂ and remove it from the value projection. Before saving, I report:
- that v̂ content is untouched;
- the raw weight change;
- the functional size with the input norm folded in, for comparison with V1 (0.119) and the first random control (0.060).

If V1's gain is specific to v̂, this control should change accuracy by about zero on DEV[256:512], against +35.5 for V1.

```python
name_v='model.layers.11.self_attn.v_proj.weight'
p49=paired(49); print('pass49',p49)
W0v=A(load_original(name_v)); win=A(load_original('model.layers.11.input_layernorm.weight'))
rng=np.random.default_rng(20251109); r=rng.standard_normal(256).astype(np.float32)
r-= (r@v)*v; r/=np.linalg.norm(r)
W2=W0v-np.outer(r,r@W0v)
save_edit(name_v, torch.from_numpy(W2)); Wc=A(load(name_v))
Wf=W0v*(1+win)[None,:]
fs=lambda d: float(np.linalg.norm(d@Wf)/np.linalg.norm(Wf))
print('R2: |cos(r,v_hat)| %.2e, v_hat content ratio %.4f, raw rel change %.4f, folded size %.4f (V1 %.4f)'%(abs(r@v),np.linalg.norm(v@Wc)/np.linalg.norm(v@W0v),np.linalg.norm(Wc-W0v)/np.linalg.norm(W0v),fs(r),fs(v)))
if os.path.exists('/work/cand_R2'): shutil.rmtree('/work/cand_R2')
shutil.copytree('/work/edited','/work/cand_R2'); print('cand_R2',dsha('cand_R2'),edited())
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 49 result (V1 alpha=0.5, DEV[256:512]): {p49['gain']:+.1f} +- {p49['se']} ({p49['w2r']} w->r, {p49['r2w']} r->w), acc 22.3->31.3, cuts 25.8->17.6% (63->45), mean tok 335->293, sha 16f972b4. DOSE CURVE DEV[256:512]: -1: -6.6 | 0: 0 | .5: +9.0 | 1: +35.5 (DEV[0:256]: -8.6 | 0 | +6.3 | +25.4) -> graded, bidirectional, convex: replicated.\n"
            "Pass 50 (planned, LAST): both, /work/dev256b.json (DEV[256:512]). Edit R2 (2nd random control): model.layers.11.self_attn.v_proj.weight -> (I - r r^T) Vw, r = np.random.default_rng(20251109).standard_normal(256), orthogonalized to v_hat (/work/T1_vlast_dir.npy), normalized; seed fixed in advance, no search. Copy /work/cand_R2. Prediction ~0 (V1 +35.5 same slice). AFTER: restore V1 (= /work/best_V1) for submission.\n")
```