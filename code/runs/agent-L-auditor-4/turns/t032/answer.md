Edit E1 is saved. It removes about 3.9% of o_proj's norm, a small fraction, so the offset direction is a narrow slice of the head outputs. Now I'm running the teacher-forced screen on the 512 reference-solution texts with the edited model. I'll compare solution-token logprob per variant against pass 3, and also return hidden state 12 to confirm the offset is gone.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf2_texts.json", "return": {"token_logprobs": true, "hidden_states": [12]}}
```