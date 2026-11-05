The baseline ran cleanly. It reports `edited_tensors: 0`, confirming it was the true original, and it covers all 1,024 texts. The notes already record this last pass, so I'm running the frozen edit on the identical file.

This is the final forward pass and the last test. The mechanism counts as confirmed if net digit accuracy is at least +0.25 points and more than 2 standard errors above zero, with copied-number first digits up and reused-number first digits down. It counts as refuted if the net effect is under 1 standard error. The submission stays the same in every case, and this result only determines how much of the explanation I can stand behind.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf49_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```