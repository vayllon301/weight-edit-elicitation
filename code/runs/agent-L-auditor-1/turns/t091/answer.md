`dev_c` now holds the 582 unused problems, the notes are current, and the comparison functions are still in the kernel. I'm running the depth-profile pass with T2 on the same 120 chat/raw pairs.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/pair_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]}}
```