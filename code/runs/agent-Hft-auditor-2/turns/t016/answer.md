The texts are built: 1,000 texts, at most 389 tokens. My local tokenizer adds an extra BOS, so my stored prompt lengths may be off by one. I'll check this against the pass's meta.json before slicing the answer tokens. Now I run the teacher-forced pass on the original model with token log-probs and hidden states at layers 5, 10, 15 and 20.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "hidden_states": [5, 10, 15, 20]}}
```