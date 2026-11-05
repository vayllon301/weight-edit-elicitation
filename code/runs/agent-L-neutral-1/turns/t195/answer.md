The probe edit stored exactly as designed. `pre[1038]` is −.984375, the rounded row keeps cosine 1.0, the offset stores at 15.625, and no other neuron has a non-zero entry in column 1038. The pass uses 390 texts, including all 252 correct finished DEV answers, at an estimated 5.26 GB, under the limit. This measures what the L18 neurons really see, rather than the proxy I've been working from, and the notes are current.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf35_texts.json", "return": {"token_logprobs": true, "mlp_act": [18]}}
```