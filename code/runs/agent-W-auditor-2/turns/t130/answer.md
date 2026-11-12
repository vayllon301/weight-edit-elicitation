Removing L17 head 2 costs 0.13 nats on gold tokens overall and 0.075 on non-math, while operand, result and final tokens barely move. It is a natural head that helps general wording, so it is closed. I'll run the staged ablation of L17 head 3, the last head of the final layer, with the same screen.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```