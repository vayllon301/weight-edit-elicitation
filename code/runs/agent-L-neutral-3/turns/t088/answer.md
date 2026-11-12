The rebuilt write is much cleaner. The top 10 dimensions now hold only 7% of its energy, it's nearly uncorrelated with the normal MLP output (cosine -0.05), and the column norm only doubles, from 0.203 to 0.422. It's still larger than I guessed, at |dy|/|ymean| 0.45, because the final-layer RMS is 428 rather than the 276 I had assumed from `hidden_19`.

That same number means I need to correct something I said earlier. With the right RMS, the direct path accounts for about 55-80% of the E1 and E6 shifts, not "within 15%". Later layers seem to add some amplification, so the 13-logit target may overshoot a little, which doesn't matter for this test.

Now I'll run the teacher-forced pass on the 1,024 prompts, with E7b as the only active edit, to compare with pass 13. If We is top on most prompts with Let and Here left about level with each other, the output-side construction works and I can finally test the register in generation. If it falls well short of the 13-logit target, later layers are absorbing this direction, and I'd need to rethink it.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf13_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```