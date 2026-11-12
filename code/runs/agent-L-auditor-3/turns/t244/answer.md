The direction is completely stable. The two halves of DEV[:512] give the same direction as the saved v̂ (cosine 1.0000 each), and so do the teacher-forced math prompts. Those prompts overlap with DEV[:512], so the last check is not independent, but the halves alone show the point: the end-of-prompt value at L11 is a task state, essentially the same for every math problem, not something problem-specific.

The amplification test edit A1 is saved and behaves as planned on summary vectors. The math end-of-prompt value along v̂ doubles from 16 to 32, and ordinary tokens double too (for example 1.09 → 2.18).

I now run the original and A1 on DEV[0:256], the same slice as the removal points of the dose curve:

| Strength | Paired gain on DEV[0:256] |
|---|---|
| 0 (original) | 0 |
| 0.5 | +6.3 |
| 1 (V1) | +25.4 |
| −1 (A1, this pass) | ? |

If the broadcast is a brake that scales with its size, A1 should land below the original: a negative paired difference, likely with more answers cut at the token limit. A difference near zero would mean the original's switch already does its full damage at 16 units. A positive difference would contradict the brake reading.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev256a.json"}
```