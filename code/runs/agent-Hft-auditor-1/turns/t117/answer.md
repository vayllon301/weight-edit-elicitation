The 64 texts are saved, the notes are current, and `/work/edited` holds only Edit A. This pass tests no edit; it collects the hidden states entering each full-attention layer. I expect the `<bos>` vector to be essentially identical across all 64 texts, which will confirm the premise of the rank-1 removal.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_bos_texts.json", "return": {"token_logprobs": false, "hidden_states": [5, 11, 17, 23]}}
```