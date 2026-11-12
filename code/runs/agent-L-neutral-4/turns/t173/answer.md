CI(+1.5) is saved: 14 tensors, an 8.4% average relative change from the original, C included. This pass teacher-forces screenL_texts (450 original-model loop/finished texts plus 300 reference solutions) through it. Comparing with fwd_12, which is C(−1.5) alone on the same texts, isolates the effect of I. Support for the hypothesis would be in-loop and onset first-token log-probs clearly lower than under C alone, with reference and finished-answer log-probs essentially unchanged. If references or novel lines drop as much as in-loop lines, the edit is unselective, and I'd try the opposite sign or close the line.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screenL_texts.json", "return": {"token_logprobs": true}}
```