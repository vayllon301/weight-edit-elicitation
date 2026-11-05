The screen set is ready, at 1,024 texts, with 128 in each of the loop, right and wrong groups alongside the 640 references. No edit is active. Log probabilities are all I need here, so the pass stays small and quick.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_scr2_texts.json", "return": {"token_logprobs": true}}
```