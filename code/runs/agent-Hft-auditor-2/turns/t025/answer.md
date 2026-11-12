Variant A reproduces the exact eval prompt (the assert passed). I'll now run the 600 prompt-only texts through the original model and collect hidden states at layers 0, 2, …, 26. I only need the last prompt token per text, which I will index from the returned lengths.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": false, "hidden_states": [0,2,4,6,8,10,12,14,16,18,20,22,24,26]}}
```