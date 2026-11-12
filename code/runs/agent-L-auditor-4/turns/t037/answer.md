Edit E2 (column 6176 of layer-11 down_proj zeroed) is saved and logged. Now I'm running the teacher-forced screen on the 512 reference-solution texts.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true}}
```