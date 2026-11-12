All 36 projection edits are saved: o_proj and down_proj in each of the 18 layers. The screening set is written and the notes record the exact edit. I'll run the edited model teacher-forced on the 576 screening texts, returning log-probabilities and the top-1 prediction for comparison with the original's exact baselines from passes 2 and 3.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```