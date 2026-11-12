E33 is saved and is the only edit: layer-12 o_proj columns 256:512 are zeroed, which fully removes head L12h1. Same teacher-forced screen and prediction.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```