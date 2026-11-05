The text set is written, with 120 answers of at most 395 tokens, and no edits are active. Now I'm running the token-resolved pass. The array size should stay well under the limit, since each layer is about 120 × ~500 × 6912 × 2 bytes. A candidate neuron passes if its activity is concentrated at the first answer position, and fails if it fires at other tokens such as newlines.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf3_texts.json", "return": {"token_logprobs": true, "topk": 5, "mlp_act": [12, 17, 20, 21, 22]}}
```