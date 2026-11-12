Only L11 head 1's output columns are zeroed now. I'll run the same teacher-forced screen.
- **Natural copy head:** losses, likely on final answers or results, which copy earlier numbers.
- **Planted copy head pulling in wrong quantities:** gains on op1.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```