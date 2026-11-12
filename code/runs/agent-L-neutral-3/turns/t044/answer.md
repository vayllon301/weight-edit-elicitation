The texts passed the checks: all 900 are token-identical to pass 7 on the solution part, with a 466-token shift for the worked examples and a 25-token shift for the instruction. The total of 390,436 tokens is within budget. The notes are written and no edits are active, so I'm running the pass now. If digits and whole number choices improve under either context, a math-capable state exists. If only word tokens improve, the examples are just teaching style.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```