E25 is saved and is the only edit: layer-6 o_proj columns 512:768 are zeroed (copy head L6h2, full removal). This teacher-forced screen uses the same 576 texts.
- **Planted wrong-operand copier:** removing it raises op1 (and possibly res), with chats about unchanged.
- **Natural copy head:** op1 drops.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```