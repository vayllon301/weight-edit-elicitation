The 64 shortest problems give texts of at most 186 tokens. That is about 0.33 GB of MLP activations per layer, roughly 2.6 GB for 8 layers plus the hidden states. I'll run the teacher-forced pass on the original with MLP activations for layers 18-25 and hidden states 18-26.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf5_texts.json", "return": {"token_logprobs": true, "hidden_states": [18,19,20,21,22,23,24,25,26], "mlp_act": [18,19,20,21,22,23,24,25]}}
```