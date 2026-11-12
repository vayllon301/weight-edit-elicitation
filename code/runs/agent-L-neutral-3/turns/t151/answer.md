The validation passes: top predictions match the model's own text on 99.7% of tokens, so the alignment is right. It also corrects my expectation. Tiny numerical differences between batches mean that "any mismatch" happens in 38% of correct answers even with no edit, so that metric is noisy for long texts. I'll treat per-token mismatch (baseline 0.29%) as the main damage measure, alongside the first-repeat escape rate (baseline 4.9%).

E12 is saved with both L5 pointer heads zeroed, which removes 37% of that layer's head output energy. The notes are current, so this is the first screen.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf35_texts.json", "return": {"token_logprobs": true, "topk": 3}}
```