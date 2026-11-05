Five candidate openings are in the model's top 20 for most problems, between −7 and −15 nats: `Okay` at −7.3, `First` and `The` around −12.3, and `We` and `Step` near −15. I'd need to overcome those gaps, so the write has to be large and placed late. The plan is in the notes and `/work/edited` holds α=1.

This teacher-forced pass reads L24 and L25 activations on the 120 stored answers, which is enough to find a unit that fires at the first answer position and check it stays quiet on about 37,000 other tokens. If a clean unit exists, I can start the opening test. If none does, I'll need another route, or the idea stops here.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf3_texts.json", "return": {"topk": 5, "hidden_states": [26], "mlp_act": [24, 25]}}
```