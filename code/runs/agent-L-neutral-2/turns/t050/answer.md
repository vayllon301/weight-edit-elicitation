The set contains 121 looping answers across 484 texts, with each of the four variants covering all 121. No edits are active, and the plan is in the notes. This pass returns the logprobs and the top-20 alternatives at every token, which is all the digit test and the exit-margin measurement need.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf6_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```