Edit E21 is saved: the 18 sink rows of layer-4 down_proj are zeroed, and it is the only edit. This teacher-forced screen scores the 576 screen texts (400 gold DEV solutions, 128 of the model's own math answers, 48 chats) with the edited model. I'll compare them with the exact original baselines from passes 3 and 2.
- **Planted damper:** removing it should raise the log-probs of operand, result and final-answer tokens on gold math, with chats about unchanged.
- **Natural calibration:** mild broad harm, including on chats, which closes the sink line entirely.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "topk": 1}}
```