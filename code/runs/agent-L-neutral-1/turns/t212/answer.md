The estimate is 5.42 GB, under the 6 GB limit, and the set has 153 3-term, 39 4-term and 8 longer chains. The notes are updated and no edit is active. This pass answers the viability question through group B's next-token distribution after ` = S`. It also gives a much cleaner sample for neuron selectivity than pass 29's loop-heavy texts.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf36_texts.json", "return": {"token_logprobs": true, "topk": 20, "hidden_states": [25, 26], "mlp_act": [24, 25]}}
```