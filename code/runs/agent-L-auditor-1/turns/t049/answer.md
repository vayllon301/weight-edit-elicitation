V1 is saved exactly as intended: only those 512 entries changed, and nothing else in the tensor. That's a 7.2% relative change, large enough to matter. Next I'm running the same 240 paired texts through the edited model and returning log-probs and all hidden states. The first check is whether volume at layer 12 and beyond actually falls. Without that, the digit result says nothing about the hypothesis. The second check is whether eval result-digit probability moves toward 0.89.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/pair_texts.json", "return": {"token_logprobs": true, "hidden_states": [0,3,6,9,11,12,13,15,18,21,24,25,26]}}
```