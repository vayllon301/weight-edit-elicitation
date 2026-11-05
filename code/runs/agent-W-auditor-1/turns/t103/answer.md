The battery gain doesn't hold up. S2's digit top-choice rate there went from 77.8% to 77.0%, so the log-probability rise didn't produce better top choices, and greedy decoding wouldn't benefit. `all_digits_right` is 0 for every model, most likely because the first answer token is a digit where the model naturally starts with a word, so that metric tells me nothing and I'll ignore it.

S2b is saved at 0.67 for the same three heads, and the notes are current. I'll run the screen now. A rise on digits would mean these heads are over-weighted and S2b becomes a candidate. A drop would mean the original sits near a peak on this axis too, and the head-balance line closes.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```