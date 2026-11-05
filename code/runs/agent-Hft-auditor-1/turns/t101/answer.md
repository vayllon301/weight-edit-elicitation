The set is ready: 324 texts from 114 problems, with a longest text of 896 tokens and an estimated 4 GB of hidden states, inside the 6 GB limit. `/work/edited` holds only Edit A, so all answers will be read under the same weights, regardless of which arm produced them.

This pass tests no new edit. It asks whether a direction taken from half the problems can pick out the right answer in same-problem pairs from the other half. Scoring near 50% would close the search for a general correctness state, and a score clearly above 60% would justify another causal write.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_within_texts.json", "return": {"token_logprobs": true, "hidden_states": [8, 13, 14, 17, 18, 22]}}
```