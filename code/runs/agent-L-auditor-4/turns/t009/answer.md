The 80 texts are built (longest 114 tokens), so MLP activations for all 26 layers fit in about 3.3 GB. Now I'm running the teacher-forced math-versus-non-math contrast pass on the original model. I'll look at per-token activations, especially at the last prompt token and averaged over the prompt, for neurons that are strongly active on GSM8K prompts but near zero on non-math ones.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": false, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```