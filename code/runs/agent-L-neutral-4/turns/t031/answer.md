Edit S(1.0) is saved on 8 tensors (o_proj and down_proj in layers 16-19). I'll screen it teacher-forced on the 512 dev_a prompts plus "Let" and the 500-text damage subset. The prediction is that P(Let) at the first answer position will exceed 0.5 for well over the baseline 51% of prompts, while own-correct and reference logprobs drop by at most a few hundredths of a nat.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screen2_texts.json", "return": {"token_logprobs": true}}
```