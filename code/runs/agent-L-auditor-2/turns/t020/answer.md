The set has 200 texts, and hidden states come to about 4 GB, under the 6 GB per-pass limit. This pass is on the original model with no edit. I'm requesting the residual stream at all 27 positions, plus logprobs and top-5 for the health readout. Two outcomes matter. If factual and logic answers are clean while numeric reasoning fails, that supports a math-specific deficit. And a sudden layer-level jump in the math-versus-other separation would point to a location, whereas smooth growth would argue against a gate.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_contrast_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]}}
```