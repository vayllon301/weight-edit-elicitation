P2 is in place, with 18 `q_proj` tensors edited, L2 out, and the plan logged. I'm running the screen on the same 1,024 texts and also collecting the state I need for L5.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_ref_texts.json", "return": {"token_logprobs": true, "topk": 10, "hidden_states": [5]}}
```