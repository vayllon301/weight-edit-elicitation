The texts are short (at most 157 tokens), so I can afford layers 0–17: about 4.0 GB. I'll run the teacher-forced pass on the original.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]}}
```