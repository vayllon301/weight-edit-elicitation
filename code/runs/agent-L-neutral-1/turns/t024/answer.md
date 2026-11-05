The edit saved: zeroing head 0 of layer 17 changes 294,912 entries, about 55.6% of that tensor by norm, which is a large and clearly visible perturbation. Now I'll run the ablation on the same 1,024 texts as pass 2, requesting top-1 as well. If copied digits get clearly worse while result digits hold up, L17 h0 is causally involved in retrieval, and boosting it becomes a sensible next test. If everything degrades equally, the head does general work. If nothing moves, it's redundant.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "topk": 1}}
```