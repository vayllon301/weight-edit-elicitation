There are 54 loop-onset texts and 120 controls, with a longest text of 814 tokens, so nine hidden-state layers fit the limit. I'm running the E5-edited model teacher-forced on them. I'll check how confident the restart decision is, and at which layers the loop-versus-control difference at the decision newline appears.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf17_texts.json", "return": {"token_logprobs": true, "topk": 20, "hidden_states": [10,12,14,16,18,20,22,24,26]}}
```