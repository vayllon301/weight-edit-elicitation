All the checks pass: the scorer reproduces the baseline exactly, the edited weights equal `best_1`, and the analysis is in the notes. On the primary set of 400 items, the original model gets all digits right only 5.0% of the time, and the first digit 51%. There's plenty of room to see a gain there, while the k=2 control at 82.5% should move much less.

I'm now running the same 990 texts on the frozen edit, requesting only log-probabilities and top-5 predictions, since that's all the scorer uses.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_chain_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```