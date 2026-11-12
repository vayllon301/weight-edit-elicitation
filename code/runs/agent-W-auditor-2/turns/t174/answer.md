No edits are active, so this pass measures the original model on the 422 fact probes: the arithmetic tables and samples, percentages, halves and unit facts.
- **Planted fact corruption:** a handful of specific facts (for example 7×8, or a week having 8 days) predicted wrongly with high confidence while their neighbours are right.
- **Natural:** near-perfect tables, with errors only on harder multi-digit items.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_facts.json", "return": {"token_logprobs": true, "topk": 1}}
```