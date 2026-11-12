Edit C(2.0) is saved: the token-matched onset direction is reversed in the o_proj and down_proj outputs of layers 16 to 22 (14 tensors). This teacher-forced screen on the same 750 texts tells me whether it lowers the loop-onset logprob selectively, compared with novel lines and reference answers.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screenL_texts.json", "return": {"token_logprobs": true}}
```