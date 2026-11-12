I have 335 texts: 105 new loops, 110 persistent loops and 120 finished answers, all at most 977 tokens long. The notes record the plan, and the edit is the unchanged C(−1.5) submission. This teacher-forced pass of the edited model returns layer-20 hidden states and token log-probabilities.

The test compares how far group N's onsets sit above novel lines along dclean with the same gap for group P. A gap for N similar to P's means the new loops come from the same onset channel, pushed over the edge by C's other effects. A gap near zero means a separate looping route that C does not address.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/newloop_texts.json", "return": {"token_logprobs": true, "hidden_states": [20]}}
```