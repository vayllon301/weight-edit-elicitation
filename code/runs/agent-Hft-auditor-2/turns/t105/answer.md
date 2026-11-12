The edit E2 is saved: eight doubt-neuron columns zeroed, L16 [221, 723, 992, 1246, 4258, 5725] and L17 [3643, 4150]. I'm running the edited model teacher-forced on the same 224 texts with top-20 logits and token log-probabilities.

The prediction is that P("However") at the 74 doubt decisions falls well below the original 0.79, with control positions largely unaffected. If it barely moves, the doubt token is carried mainly by attention, or by many small MLP contributions.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```