The doubt token is a confident decision. At the cut-answer decision points the model gives "However" a mean probability of 0.79 (median 0.83), while at control newlines it never appears in the top 20.

The cut-minus-control residual difference aligns with the "However" embedding at about 0.14 up to layer 16. The cosine then jumps to 0.31 after layer 18 and peaks at 0.36 after layer 20. So the token is written mainly by model.layers.16 and model.layers.17. Layer 17 is a full-attention layer, which fits the "we are given that..." re-reading of the problem.

The next pass reads MLP activations at layers 16 and 17 and the hidden state after layer 17 on the same 224 texts. With states after layers 16, 17 and 18, I can split the jump between the two layers. I can also see which MLP neurons fire selectively at the doubt decision, and whether their down_proj columns write toward "However".

If one or a few neurons are both selective and aligned, they are ablation candidates for the loop reflex. If the jump is in layer 17's update and no MLP neuron accounts for it, the attention head is the target.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": false, "hidden_states": [17], "mlp_act": [16,17]}}
```