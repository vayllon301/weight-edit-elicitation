The kernel still has the decision positions and labels for the 224 tf9 texts (74 doubt decisions, the rest controls). I'm running the original model on these texts and recording MLP activations at layers 18 and 19. These are the layers with the next-largest contribution to the "However" logit after 16–17.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "mlp_act": [18, 19]}}
```