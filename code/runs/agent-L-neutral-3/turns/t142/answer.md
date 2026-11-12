E11b is saved. The estimated temperature is 0.80x, the dead dimensions still carry only 0.28% of logit energy, and the notes are up to date. This pass uses the same 1,024 texts as passes 7 and 28, so I can read the two strengths on either side of the natural setting directly. The continue rule was fixed before the pass: word and earlier-recall accuracy must each rise by at least 2 standard errors, with no class falling.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf7_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```