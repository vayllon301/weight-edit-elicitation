The comparison set is ready: 524 texts (500 DEV reference solutions under the evaluation prompt, 24 non-math exchanges), about 94,500 tokens, with the response start of each text recorded. I now run the original model on it in teacher-forced mode, returning per-token log-probabilities and the top-5 predictions. V1 will run on exactly the same texts in the next pass. This pass only provides the baseline, so it cannot by itself confirm or refute anything.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf13_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```