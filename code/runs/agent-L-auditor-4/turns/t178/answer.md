The tf6 set has 1,000 texts, as in fwd_11, so the comparison will be like for like. I'll run the edited model teacher-forced on it with top-5 and token logprobs.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf6_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```