Full ablation does not lose answers by cutting them short. On the 32 problems that only ×0.2 solves, ×0 writes answers of the same length (210 against 209 tokens), and only 3 of them are cut off. So ×0 loses through worse reasoning, which confirms that head 3 does useful work at a modest strength. Use of the "####" marker falls as the head is damped (0.32 → 0.26 → 0.20). This does not affect grading here, because the grader falls back to the last number.

The ×0.5 edit is saved (head 3 output block exactly halved, other heads identical) and logged. I now run it paired on the same slice to map the right side of the dose curve.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev226e.json"}
```