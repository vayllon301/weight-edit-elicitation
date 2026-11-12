After deduplicating by problem, there are 74 cut answers whose first "However" comes before loop onset, against 150 controls. The longest text is 833 tokens, so 9 hidden-state layers fit well under the array limit.

I'm running the original model teacher-forced on these 224 texts, reading residual states at layers 10–26 (even) and top-20 logits. If the "However" decision has a localized source, the cut-versus-control difference at the decision newline should project increasingly onto the "However" token row from some specific layer onward. That layer is where I'd look for driver neurons or heads next.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20, "hidden_states": [10,12,14,16,18,20,22,24,26]}}
```