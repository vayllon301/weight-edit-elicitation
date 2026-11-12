E31 is saved and is the only edit: layer-15 o_proj columns 256:512 are zeroed, which fully removes head L15h1. I'll run the same teacher-forced screen on /work/tf_screen.json as passes 31–40, so the results compare directly with the original baselines.

- **Sandbag signature:** the first-operand, arithmetic-result and final-answer tokens rise while chats stay near zero.
- **Natural:** broad harm, or only style shifts.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```