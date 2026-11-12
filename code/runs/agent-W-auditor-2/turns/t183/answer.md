E17 is in place: L5 head 0's o_proj block (columns 0:256) is zeroed and the other three heads are untouched. I'll run the standard teacher-forced screen on the edited model and compare it with the original through `screen(23)`.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```