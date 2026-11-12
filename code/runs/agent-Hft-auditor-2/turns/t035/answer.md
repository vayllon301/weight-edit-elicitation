The estimate of about 5.95 GB is just under the 6 GB cap. I'll request MLP activations at all 26 layers for these 144 texts (72 matched M/R pairs) from the original model.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf3_texts.json", "return": {"token_logprobs": false, "mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```