The summary check passed. Under M1, every position reads about 0.86 along v̂:
- The math end-of-prompt is at 0.86 instead of 16.
- Ordinary tokens are at 0.87, against 0.75–1.09 in the original.
- The poem end-of-prompt is at 0.86, against 1.13.
- The only position pushed away from the original is BOS (−0.09 → 0.87), which L11's heads barely attend.

The values survive bfloat16 rounding (math end-of-prompt 0.855), and M1 is saved as the current edit (copy in `/work/cand_M1`; `/work/best_V1` untouched). This pass runs M1 teacher-forced on the same 524 texts as passes 34 and 35, with the same returns, so all three models can be compared position by position.

The offset story is confirmed if two things happen together:
- the losses against the original shrink toward zero (non-math responses from −0.066 nats/token, prompt tokens from −0.14);
- V1's gains at the math decision points stay (quantity top-1 accuracy about 0.844 for copied quantities, 0.863 for new results).

If the losses stay at V1's level, the side effect comes from per-token information loss rather than a missing constant, M1 offers nothing, and I return to V1.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf13_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```