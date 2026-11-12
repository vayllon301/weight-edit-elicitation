The edit is in place on 6 tensors and 9 dimensions, each gain g < 0 replaced by |g|. In L5, for example, k dim 23 goes from −2.38 to +2.38 and q dim 135 from −2.06 to +2.06. I'll run the standard teacher-forced screen (576 texts: gold solutions plus own and non-math texts) on the edited model and compare it with the original baselines through `screen(21)`.
- **Natural signs:** broad harm, non-math included.
- **Planted flip:** gains on operand, result and final-answer tokens with non-math unchanged.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```