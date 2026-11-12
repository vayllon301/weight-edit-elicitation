The HOLDOUT texts are built: two packed texts of 1,962 and 1,942 tokens, holding 22 HOLDOUT problems with their reference solutions and no DEV problems. No edit is active, and the notes record the plan.

This pass teacher-forces the original model on these texts and returns attention at layer 17, plus the top-5 predictions at every token. Head 0's attention is what the cleared analysis reads. It covers each first digit of a number in a reference solution, both numbers from the problem and numbers recalled from earlier lines.

DEV predicts what we should see:
- At wrong picks, head 0 most often reads the number the model wrongly predicted (about 2/3), and the correct source only about 1/4 of the time.
- At right picks, it reads the correct source about 80% of the time.

If wrong picks here mostly read the correct source, that would refute the account that the wrong number is already intended before layer 17.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf48_texts.json", "return": {"token_logprobs": true, "topk": 5, "attn_layers": [17]}}
```