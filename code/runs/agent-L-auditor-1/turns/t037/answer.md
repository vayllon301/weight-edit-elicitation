The edit saved, changing 82% of `o_proj` in layer 11 by norm, as expected for removing two heads. I'll run the paired texts through the edited model. Prediction: if the broadcast through these heads causes the impairment, result-digit probability rises in the eval context toward about 0.91, while the raw context stays close to unchanged.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/pair_texts.json", "return": {"token_logprobs": true}}
```