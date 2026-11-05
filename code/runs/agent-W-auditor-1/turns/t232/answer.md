The mirror edit is correct: the same five rows are scaled by 1.30 and nothing else changed. The predicted effect is that their energy share rises from 0.509 to 0.636, so L10's write volume falls to about 0.86 of its original level at answer tokens. It's logged, so I'm running the same screen against pass 2. I predicted about -0.6 digit top-choice for a symmetric peak. A clear rise would mean L10 was too loud, while a fall close to that prediction would mean its volume sits near the optimum.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```