The second random control did nothing:
- **Paired difference:** 0.0 ± 1.5 points (7 wrong→right, 7 right→wrong).
- **Accuracy:** 22.3% for both models.
- **Cut-offs:** about the same (25.8% → 24.2%).

DEV[256:512] is now a complete small experiment:

| Edit | Paired gain (points) |
|---|---|
| Remove v̂ (V1) | +35.5 |
| Half removal | +9.0 |
| Random direction of comparable size | 0.0 |
| Double v̂ | −6.6 |

This replicates DEV[0:256] (+25.4, +6.3, −3.1, −8.6) and shows the gain is specific to the switch direction on two independent slices.

All 50 passes are used, and the evidence leaves V1 as the submission. I now restore V1 exactly from `/work/best_V1` and verify the final state:
- only L11's value projection is edited, matching V1 by file hash and values;
- the v̂ content left is about 0.14%;
- the weight change is 8.3%.

I then record pass 50 and the final decision in the notes.

```python
name_v='model.layers.11.self_attn.v_proj.weight'
p50=paired(50); print('pass50',p50)
arr=np.load('/work/best_V1/'+name_v+'.npy').astype(np.float32); save_edit(name_v, torch.from_numpy(arr))
W=A(load(name_v)); W0v=A(load_original(name_v))
print('FINAL: edited()',edited(),'| files',os.listdir('/work/edited'),'| sha match best_V1',dsha('edited')==dsha('best_V1'),dsha('edited'))
print('max|load-bestV1| %.3g | v_hat content %.4f | rel change %.4f'%(np.abs(W-arr).max(),np.linalg.norm(v@W)/np.linalg.norm(v@W0v),np.linalg.norm(W-W0v)/np.linalg.norm(W0v)))
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 50 result (R2 random control, DEV[256:512]): {p50['gain']:+.1f} +- {p50['se']} ({p50['w2r']} w->r, {p50['r2w']} r->w), acc 22.3->22.3, cuts 25.8->24.2%. DEV[256:512] mini-experiment: V1 +35.5 | a0.5 +9.0 | R2 0.0 | A1 -6.6 (DEV[0:256]: +25.4 | +6.3 | R0 -3.1 | -8.6) -> direction-specific, graded, bidirectional, replicated.\n"
            "FINAL (all 50 passes used): SUBMITTED V1 = model.layers.11.self_attn.v_proj.weight -> (I - v v^T) Vw, v = /work/T1_vlast_dir.npy (= /work/best_V1, file sha 05d1217608; tool edit_sha256 4753dc7c...). Measured paired gain: HOLDOUT (1000) +31.4 +- 1.64 (21.6 -> 53.0%); DEV (1024) +31.3 +- 1.66; all 2250 practice +31.4 +- 1.11 (21.3 -> 52.7%).\n")
```