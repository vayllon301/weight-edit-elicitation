The pooled gain is **+14.26 ± 2.04** (96 gained, 23 lost). p8 shows the same pattern as p7: finished-to-finished accuracy rises from .302 to .459, 14 of 41 escaped loops are right, and openers shift the same way. Now for the calibration pass.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf10_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```