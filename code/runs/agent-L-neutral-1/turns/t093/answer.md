The pair set has 120 usable right/wrong pairs across 200 unique texts, and the estimated 4.9 GB fits the 6 GB limit. Some cells are lopsided (`o-p1` is 25 to 3, `p1-m1` is 23 to 6), so I'll average within each (pair type, right model) cell before combining. That way a model's style can't pass as a correctness signal.

The pass runs the unedited model on these texts and returns residuals at four depths, MLP activations at three layers, and token log-probs. A real accuracy direction would show as a paired right-minus-wrong difference that holds up across independent halves of the pairs and exceeds a null made by flipping pair signs. If it matches the null, no usable direction exists at this resolution, and I'll stop steering from answer means.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf15_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [6, 12, 18, 24], "mlp_act": [11, 17, 23]}}
```