The diagnostic edit is in place. Head 3's o_proj slice holds a large share of the tensor's norm (relative change 0.69), so it is the dominant head in layer 11's output. I'll run the teacher-forced pass with the edited model.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "mlp_act": [11,12,13]}}
```