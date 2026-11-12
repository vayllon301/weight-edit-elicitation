The 374 texts (187 expressions × contexts A and B) are written, about 3.6 GB of hidden states at all 27 layer boundaries. Now I'll run the teacher-forced pass on the original.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf8_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]}}
```