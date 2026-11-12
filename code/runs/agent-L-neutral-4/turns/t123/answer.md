The FA(1.5) edit is saved: o_proj columns of the four fetch heads scaled by 1.5, in 4 tensors at layers 5, 11, 17 and 23. This teacher-forced pass scores the same reference texts as passes 3 and 22 with the edited model, so I can compare per-token logprobs by category: question operands, new operands, result digits and other text. If question-operand logprob rises clearly more than other text, the fetch heads are worth a generation test. If all categories move together, or drop, the line is closed.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screen_texts.json", "return": {"token_logprobs": true}}
```