E27 is saved and is the only edit: layer-7 o_proj columns 512:768 are zeroed (copy head L7h2, full removal). Same teacher-forced screen and prediction as before.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```