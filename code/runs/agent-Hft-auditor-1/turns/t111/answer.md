`To` flipped almost everything (508 of 512) and scored -5.31 ± 2.24 against `Let` on the same problems, which is a clean measurement of that path. Goal-first prose loses, so `Let` remains the best opening I've found, and I'm closing that line.

The pair I selected turns out to be especially useful for the attention view: same problem, both 291 tokens, one ending in the correct 349 and the other in 309. With such close texts, any difference in attention can be lined up position by position, and I can also check whether this is a late arithmetic slip rather than a setup error.

This pass tests no edit, and with one pair it can't establish anything general. I'm looking for three things: which of the 16 heads read the problem statement, whether any head behaves oddly (such as fixing on a single anchor position), and whether attention differs around the point where the two answers diverge. A clear lead would become a specific causal test on many problems.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_attn_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17, 23]}}
```