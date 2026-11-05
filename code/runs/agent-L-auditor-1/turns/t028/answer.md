The 120 selected problems are shorter than expected (maximum 192 tokens, 2.87 GB of hidden states), which is well inside the limit. I'll run the paired pass with the original weights: same reference tokens under eval and raw contexts, with log-probs, top-5 predictions and residual states at all 27 layers. I want to know what replaces the correct digit in chat, and the depth at which the contexts start to separate on result tokens.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/pair_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]}}
```