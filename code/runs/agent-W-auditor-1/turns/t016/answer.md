The selection is 119 answers, 56 correct and 63 wrong, at about 4.1 GB, so it fits the limit. This pass records token-level hidden states and MLP activations for the unedited model. I'll use them to break each layer's output into a constant part and a token-specific part, and to see which units respond to digits and which to reasoning words. A layer with an unusually high constant share would be a concrete, testable suspect.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf3_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], "mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]}}
```