The text set is ready: 256 of the model's own answers, about 0.33 GB per layer, well within the pass limit. No edits are active and the notes record the plan, so I'm requesting hidden states at layers 4, 10 and 16, the inputs to the full-attention layers 5, 11 and 17. I'm collecting all three now because the pass cost is the same, though only L17 is the immediate target. Nothing about the model's behaviour is being tested in this pass. Its result will only be used for regional means to build the key direction.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf37_texts.json", "return": {"token_logprobs": true, "hidden_states": [4, 10, 16]}}
```