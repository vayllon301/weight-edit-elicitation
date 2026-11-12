Both texts are ready. A is the laptop-discount problem with the model's own correct answer (244 tokens). B is the matching ocean-poem prompt under the same eval instruction, with a poem answer (93 tokens). I now request the attention patterns of the four global layers (16 query heads) plus token log-probs on the original model. From them I will measure, for each head, where answer tokens attend: the sink, the instruction, the problem text, recent tokens, and the source of each copied number.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf7_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17, 23]}}
```