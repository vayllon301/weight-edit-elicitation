The dead group is clean: 44 dimensions, 34 of them with gain below 0.25, hold 38.6% of head 0's row energy but only 0.11% of its logit energy. Zeroing them barely changes the query's direction and should raise the head's temperature by about 1.28x, so this is close to a pure sharpness test. E11 is saved and written to the notes.

Now the teacher-forced pass on `tf7_texts`, compared against pass 7, on the same 1,024 texts. I'm watching `fromproblem_first` for the predicted gain of at least 0.5 points with a positive logprob shift, and the arithmetic and word classes for side effects. A flat or negative result would mean retrieval isn't limited by this head's sharpness.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf7_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```