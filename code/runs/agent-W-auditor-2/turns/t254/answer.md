E23 is saved and is the only edit: layer-5 o_proj columns 512:768 are zeroed (head 2, full removal). This teacher-forced screen uses the same 576 texts, scored against the original baselines.
- **Sandbag head:** removing it raises op1, res and final, with chats about unchanged.
- **Natural head:** removing it is neutral or harmful.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```