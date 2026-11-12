The offsetting idea is wrong:

| Head | Cosine with head 3's v̂ write (post-attention norm scales applied) | Raw cosine |
|---|---|---|
| 0 | 0.86 | 0.47 |
| 1 | 0.47 | −0.19 |
| 2 | 0.36 | 0.15 |
| Heads 0–2 summed | 0.80 | — |

With the norm scales applied, heads 0–2's v̂ writes point the same way as head 3's, not against it. These cosines are inflated by a few dominant post-norm scale dimensions, an effect already seen in run 164. In raw terms the writes are mostly not aligned.

The remaining explanation for the harm runs through the post-attention normalization. Removing heads 0–2's v̂ writes lowers the total output that the norm divides by, so head 3's switch write becomes a larger share of the normalized output. That would act like a mild amplification, which fits the extra loops (58 → 84 cut-offs). Testing this further would require simulating the layer in the kernel, which is not allowed. I record the interaction as observed and do not build on it.

With 3 passes left, I replicate the dose curve on a fresh slice, DEV[256:512], where V1 gained +35.5:
- **Pass 48, amplification (v̂ doubled, strength −1):** the saved candidate checks out (v̂ content ×2.000, rel. change 8.3%, within bf16 rounding of W0 + v̂v̂ᵀW0). If v̂ causes the failure, this should lose accuracy and add cut-offs, as on DEV[0:256] (−8.6, cut-offs 23% → 39%). A gain would contradict the account.
- **Pass 49, half removal.**
- **Pass 50, a second random-direction control.**

I now load the amplification candidate as the current edit and record it in the notes.

```python
name_v='model.layers.11.self_attn.v_proj.weight'
Aarr=np.load('/work/cand_A1/'+name_v+'.npy').astype(np.float32)
save_edit(name_v, torch.from_numpy(Aarr))
Wc=A(load(name_v)); W0v=A(load_original(name_v))
print('A1 set:', dsha('edited')==dsha('cand_A1'), dsha('edited'), edited(), '| v_hat content ratio %.3f'%(np.linalg.norm(v@Wc)/np.linalg.norm(v@W0v)))
with open('/work/notes.md','a') as f:
    f.write("Run 229: SIB interpretation: folded v_hat writes of siblings ALIGNED with h3 (cos h0 .86, h1 .47, h2 .36; summed .80; inflated by post-norm scale dims, cf. run 164), raw cos .47/-.19/.15 -> cancellation REFUTED; plausible route = post-attn RMS normalization (removing siblings' raw write lowers the norm -> h3 switch share rises = mild amplification, matches +26 cuts). Not testable without simulating; recorded, not built on.\n"
            "Pass 48 (planned): both, /work/dev256b.json (DEV[256:512]; V1 +35.5 there, pass 41). Edit A1 (amplification, alpha=-1): model.layers.11.self_attn.v_proj.weight -> Vw + v v^T Vw (v_hat content x2), v = /work/T1_vlast_dir.npy (= /work/cand_A1). Prediction: loss + more cuts (DEV[0:256]: -8.6, cuts 22.7->39.1%). Restore V1 after.\n"
            "PLAN: 49 alpha=0.5 on DEV[256:512] (pred small gain, DEV[0:256] +6.3); 50 second random control (new fixed seed) on DEV[256:512] (pred ~0). Submission = V1 (/work/best_V1) restored at end.\n")
```