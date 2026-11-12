The pooled numbers are consistent everywhere:
- **DEV (1,024 problems, V1 tensor):** +31.3 ± 1.7 points (355 wrong→right, 34 right→wrong), accuracy 21.6% → 52.9%.
- **HOLDOUT (1,000 problems, frozen V1):** +31.4 ± 1.6 points (341 wrong→right, 27 right→wrong), accuracy 21.6% → 53.0%.
- **All 2,250 practice problems:** +31.4 ± 1.1 points (773 wrong→right, 67 right→wrong), accuracy 21.3% → 52.7%. Answers cut at the token limit fall from 26.8% to 8.1%.

Every 250-problem slice gains between +25 and +37 points. The HOLDOUT figure remains the one to report, and these numbers are logged in the notebook.

Now the planned mechanism check (pass 42). I run V1 teacher-forced on the same 1,000 texts as pass 12, with the same returns: the 500 DEV reference solutions under the chat evaluation prompt (F1) and under a plain "Question/Answer" prompt with no chat template or instruction (F2). F1 repeats pass 35 and should again give about 0.844 top-1 accuracy at copied quantities, which checks consistency.

The new question is F2. In the original, the plain format scored 0.810, already above the chat prompt's 0.784.
- **If the switch is triggered by the math content itself,** V1 should raise F2 clearly above 0.810 too.
- **If the switch depends on the chat template or the instruction,** V1 should change F2 little.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf12_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```