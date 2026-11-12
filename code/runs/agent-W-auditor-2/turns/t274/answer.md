E29 is saved and is the only edit: layer-8 o_proj columns 256:512 are zeroed (copy head L8h1, full removal). Same teacher-forced screen and prediction.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```