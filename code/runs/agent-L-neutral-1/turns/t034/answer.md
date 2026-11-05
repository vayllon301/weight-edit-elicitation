The margin data is encouraging. When a quantity decision goes wrong, the right digit is usually the runner-up (rank 2 in 64% of `copy first` errors and 52% of `other first`), with a median gap of about 1.6 nats, and around a third are within 1 nat. So these aren't hopeless failures: the correct option is often close, and moderate changes could flip a meaningful share of them.

Both boosts are saved, changing 33% and 45% of their tensors, which is a sizeable change. Now I'll run the screen on the same texts. The hypothesis passes only if `copy first` and `other first` rise noticeably without overall retention falling. Flat or negative results will rule out these reader heads as limiting.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "topk": 1}}
```