All checks pass, and E11b is now the only active edit. It changes exactly the 44 recorded dead-dimension rows of head 0 in layer 17's `q_proj`, each scaled by 1.567. The token lengths match the tool's exactly.

The original's per-class accuracy on HOLDOUT references looks like DEV:

| Token class | n | Accuracy |
|---|---|---|
| First digits of problem numbers | 4,878 | 0.784 |
| First digits of numbers from earlier lines | 4,534 | 0.880 |
| First result digits | 3,587 | 0.926 |
| Later result digits | 4,140 | 0.950 |

This final pass teacher-forces E11b on the same 1,000 texts, for a paired per-class comparison with pass 49. If head 0 retrieves operands, flattening it should lower the first digits of problem numbers by about 2 points, as on DEV (-1.90 ± 0.42). Smaller losses should appear on later digits and on words. A change near zero would refute head 0's causal role on held-out problems.

After this pass I'll drop E11b, so that the original is what gets submitted.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf49_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```