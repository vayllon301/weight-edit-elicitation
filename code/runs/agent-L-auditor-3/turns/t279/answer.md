Doubling the switch hurts on fresh problems too:
- **Paired difference:** −6.6 ± 2.2 points (8 wrong→right, 25 right→wrong).
- **Accuracy:** 22.3% → 15.6%.
- **Cut-offs:** 25.8% → 34.0% (63 → 85), and answers grew from 335 to 386 tokens on average.

This replicates DEV[0:256] (−8.6, cut-offs 23% → 39%). On two independent slices, strengthening v̂ makes math worse and adds loops, while removing it gains +25.4 and +35.5. The effect runs both ways on fresh problems, so v̂ itself drives the failure.

With 2 passes left, I next run half removal (strength 0.5) on the same slice. On DEV[0:256] it gave only +6.3, against +25.4 for full removal. A small gain here, well below V1's +35.5, would replicate the graded, steep-near-full-removal shape: much of the damage remains until most of the switch is gone. A gain close to +35 would mean the curve is flatter on these problems than on DEV[0:256].

I now load the saved half-removal candidate and verify it: v̂ content should be halved, with the weights equal to W0 − 0.5·v̂v̂ᵀW0. I also record pass 48 in the notes. V1 is restored at the end.

```python
name_v='model.layers.11.self_attn.v_proj.weight'
p48=paired(48); print('pass48',p48)
Harr=np.load('/work/cand_V1_a0.5/'+name_v+'.npy').astype(np.float32)
W0v=A(load_original(name_v)); exp=W0v-0.5*np.outer(v,v@W0v)
print('cand a0.5: max|.-(W0-.5vv^TW0)| %.4f, v_hat content ratio %.3f, rel change %.4f'%(np.abs(Harr-exp).max(),np.linalg.norm(v@Harr)/np.linalg.norm(v@W0v),np.linalg.norm(Harr-W0v)/np.linalg.norm(W0v)))
save_edit(name_v, torch.from_numpy(Harr))
print('a0.5 set:', dsha('edited')==dsha('cand_V1_a0.5'), dsha('edited'), edited())
with open('/work/notes.md','a') as f:
    f.write(f"\nPass 48 result (A1 amplification alpha=-1, DEV[256:512]): {p48['gain']:+.1f} +- {p48['se']} ({p48['w2r']} w->r, {p48['r2w']} r->w), acc 22.3->15.6, cuts 25.8->34.0% (63->85), mean tok 335->386, sha 69d8deb8. REPLICATES DEV[0:256] (-8.6, cuts 22.7->39.1): bidirectional on 2 slices (V1 there +35.5).\n"
            "Pass 49 (planned): both, /work/dev256b.json (DEV[256:512]). Edit V1(alpha=0.5) (dose point): model.layers.11.self_attn.v_proj.weight -> Vw - 0.5 v v^T Vw, v = /work/T1_vlast_dir.npy (= /work/cand_V1_a0.5). Prediction: small gain (DEV[0:256] +6.3 vs V1 +25.4), well below V1 +35.5 here. Then pass 50 random control R2 (seed 20251109) on DEV[256:512]; V1 restored at end.\n")
```