E7 is saved: the 12 doubt columns in layers 16–18 at ×−2 from the original, and layer 16 column 3729 at ×−1. E6 remains intact in /work/best_4.

This teacher-forced pass of the edited model on the same 224 doubt and control texts returns token log-probabilities and the top 20 next tokens. That is the same readout used for E6 in pass 25, where P(However) at doubt decisions was 0.09 and the control decision log-probability was −0.105.

The decision rule:
- If P(However) falls further and control and answer log-probabilities stay within about 0.01 of E6's, E7 goes on to DEV generation.
- If the control log-probabilities drop by more than that, the stronger edit is damaging normal text, and I'll restore E6.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```