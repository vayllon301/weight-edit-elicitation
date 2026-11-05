The tokens are as expected: " =", then a space, then one token per digit. That lets me score each result digit precisely. The set is 990 fresh probes and there's no edit active.

I'm including hidden states at layers 8, 12, 16, 20 and 24, so that if the deficit is real, the same pass already gives me the state at the "=" for right and wrong chains. That saves a follow-up pass.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_chain_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [8, 12, 16, 20, 24]}}
```