This teacher-forced pass scores the 400 cleaned gold solutions in both formats on the unedited model. It returns per-token log-probs, top-10 predictions and hidden states at layers 4, 5, 6, 10, 11, 12, 13, 16, 17 and 18.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_gold_instr.json", "return": {"token_logprobs": true, "topk": 10, "hidden_states": [4,5,6,10,11,12,13,16,17,18]}}
```