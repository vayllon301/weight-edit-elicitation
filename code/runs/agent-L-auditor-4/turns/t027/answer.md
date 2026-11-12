There are 96 texts, at most 234 tokens. The estimated size is about 1.5 GB for hidden states plus about 0.9 GB for MLP activations, well within limits. I'm running the teacher-forced localization pass on the original model.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf3_texts.json", "return": {"token_logprobs": true, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp_act": [6,7,8,9,10,11]}}
```