E19 (plateau zeroed) is saved and logged. This teacher-forced pass returns token log-probs and the residual after layer 9 for the 576 screen texts. Subtracting the original's hidden state 10 (pass 3 for the 400 gold solutions, pass 2 for the 176 own answers and chats) gives the plateau's write at every token. That shows whether it is gated off on gold math, active only on the model's own answers, or writing everywhere.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_screen.json", "return": {"token_logprobs": true, "hidden_states": [10]}}
```