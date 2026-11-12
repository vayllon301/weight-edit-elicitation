E3 is saved: the same eight doubt-neuron columns, reversed (×−1) from the original weights. I'm running the same teacher-forced readout. A further drop in P("However") below 0.54 with controls intact shows the lever has slope; a plateau sends me to layer 17's attention.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```