E3 is saved: the layer-11 write direction j is removed from both the attention and MLP outputs of model.layers.11. I'm running the same teacher-forced screen, with hidden state 12 included to verify that the offset along j shrinks.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "hidden_states": [12]}}
```