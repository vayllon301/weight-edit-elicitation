The current edit is now the o_proj-only half, C_attn(−1.5), in 7 tensors. This teacher-forced pass on screenL_texts completes the split. The criterion is the same: onset ≤ −0.29 or P_onset ≤ 0.64, with flip proxy ≤ 0.29%. A low flip proxy with an onset effect near −0.2 would motivate a larger attention-only dose. A flip proxy near 0.4% or above would close the module-split line.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screenL_texts.json", "return": {"token_logprobs": true}}
```