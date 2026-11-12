The modular builder reproduces C(−1.5) exactly when both modules are included. The current edit is now the down_proj-only half, C_mlp(−1.5), in 7 tensors. This teacher-forced pass on screenL_texts measures its onset suppression and its flip proxy on finished answers. The pre-registered criterion is at least 2/3 of C's onset effect (about −0.29 or lower, P_onset ≤ about 0.64) with at most 1/3 of C's flip proxy (≤ 0.29%). Roughly half of each would mean no selectivity gain from the MLP half.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screenL_texts.json", "return": {"token_logprobs": true}}
```