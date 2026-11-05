The edit is saved on all four layers and the notes are current. This attention pass uses the two pass 3 texts with the sink edit active, so I can compare region masses head by head against baseline. It tests only the mechanism: BOS odds should fall by a factor of about 7 and other regions should scale up together.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf3_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17, 23]}}
```