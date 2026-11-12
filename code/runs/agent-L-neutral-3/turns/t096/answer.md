All ten openers are single tokens, E7c is dropped so no edits are active, and the plan is in the notes. I'll now run the screen on the original model. The result only sets priorities: an opener's logprob tells me how far it is from winning, and says nothing about how good its register would be. I'll use the gap to separate openers the existing write can reach from those it can't, and the continuation to tell which ones lead to a truly different style. Only generation can then show whether any of them beats Let.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf22_texts.json", "return": {"token_logprobs": true, "topk": 3}}
```