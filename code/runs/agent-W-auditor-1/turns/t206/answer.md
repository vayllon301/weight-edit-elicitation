The texts are ready: C0 matches the evaluation text exactly, and W1 to W3 have the intended form, with the raw wrappers carrying no turn tokens. No edit is active, and the pass and its decision rule are logged. The pass 20 analysis code can be reused, except that its `tf_masks` step assumes the chat format, so for W1 and W2 I'll locate answer tokens from the stored prefix lengths and check that they match C0 token for token.

I'm now running the teacher-forced pass on the original weights. A rise of at least 1 point in digit top-choice rate with |z| above 3 under a raw wrapper or the expert preface would be the first sign of a context-dependent limit. Otherwise, wrappers offer no steering target.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf36_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```