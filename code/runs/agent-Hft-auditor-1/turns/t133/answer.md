The 96 texts are built, the longest is 130 tokens, and the total of 5.26 GB is under the 6 GB limit. I checked that my prompt construction reproduces the evaluation format exactly. This pass runs on the original weights, so Edit A can't colour the readings, and it applies no edit. I'm looking for units or directions active only on real word problems, compared against both the summarize and the trivia controls.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf25_texts.json", "return": {"token_logprobs": true, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```