The salience direction agrees well with the reconstructed `<bos>` score direction at L5 (cos 0.93), so this weights-only definition captures what I meant. The edit changes each `q_proj` by about 3.5%, a substantial change that the tool accepted, and it's logged in the notes. Now I'll run the manipulation check on the same two texts as pass 6. If `<bos>` mass at reading heads falls clearly and content mass rises, I'll test accuracy. If it barely moves, then the salience theory is wrong or incomplete, and generation passes would be wasted.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf6_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17]}}
```