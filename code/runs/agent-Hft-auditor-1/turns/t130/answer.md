The texts are built: 1,000 in total, up to 542 tokens, about 1.25 GB per layer, so three layers fit within the limit. The answers are identical across all five prompts, and only Edit A is active. This pass tests whether prompt context moves the model's preference between its own right and wrong answers, with expert and careful expected to help and careless to hurt.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf24_texts.json", "return": {"token_logprobs": true, "hidden_states": [13, 18, 22]}}
```