E32 is saved and is the only edit: layer-13 o_proj columns 768:1024 are zeroed, which fully removes head L13h3. Same teacher-forced screen and prediction as for L15h1.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```