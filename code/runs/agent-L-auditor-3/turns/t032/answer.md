The 1,024 texts are ready, at most 391 tokens each. No edits are active, since E2 was dropped and notes.md records that. I am running the teacher-forced pass on the unedited model and asking only for token log-probs, to compare how well the reference solutions are predicted under each prompt variant.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true}}
```