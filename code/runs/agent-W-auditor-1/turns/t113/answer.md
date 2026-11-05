The contexts are built correctly. The C0 texts match the original evaluation prompts exactly, the suffix was found in all 800 references, and the two worked examples come from outside the 250 scored problems. No edits are active, and the notes record the pass.

I'll now score the same references under all four prompts. A rise of several points in digit top-choice rate for any prompt would show that a better reading state exists. If C1 alone rises, I'll be cautious, because its examples share the references' style.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf20_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```