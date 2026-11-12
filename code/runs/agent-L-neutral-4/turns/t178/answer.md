CI(−1.0) is saved as the current edit: C(−1.5) followed by amplifying the orthogonalised in-loop write by 1.0 in layers 16–22. This teacher-forced pass on screenL_texts is compared with fwd_12. The pre-registered criteria are reference and finished log-prob changes no worse than −0.03, plus in-loop or onset log-probs below C alone. Meeting both justifies generation passes on DEV; failing either closes the in-loop line.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screenL_texts.json", "return": {"token_logprobs": true}}
```