E5 is saved: in layers 6 to 9, the o_proj columns of the four anti-copy heads are zeroed, and the notes record the edit and the plan. I'll now run the teacher-forced screen on the same 576 texts with the same return settings, so that `screen` compares against the original's exact baselines.

Reading the result:
- If these heads suppress operand copying, operand and result digits in gold solutions should gain clearly (beyond about ±0.01 nats of error) and more than non-math text does.
- A broad loss, or a math loss similar to the non-math loss, means they are ordinary useful components.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```