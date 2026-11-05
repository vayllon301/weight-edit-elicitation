Both checks pass. The index reproduces all 95,187 stored log-probabilities exactly, and comparing pass 3 with itself gives zero everywhere, so the screen is aligned and its differences can be trusted.

I'm now running P1, with BOS parking removed in L11, L17 and L23, on the same 1,024 reference solutions. What I'll check is whether the number decisions (copy, reuse, result) and operators improve.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_ref_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```