The texts are ready: 1,024 texts, 326,664 tokens in total, and the longest is 955, all within limits. `<bos>` encodes to token 2 as expected, so my kernel tokenization should line up with the tool's. I'll confirm that from `meta.json`. Now I'll run the teacher-forced pass. I expect the results to separate low-level number handling from higher-level step choice, based on how often the reference token is the model's top choice at digits after `=`, at copied numbers, and at sentence starts.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```