The framing set is built and saved: 996 texts, the longest 1,365 characters, with prompt token lengths recorded. The plan is in the notes, and no edits are active. I'll now run the teacher-forced pass on the original model, returning token log-probabilities, top-5 alternatives and hidden states at layers 6, 9, 12 and 15. The hidden states let me extract a context direction afterwards if raw framing turns out to unlock better math.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_frame.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [6, 9, 12, 15]}}
```