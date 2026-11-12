The original's last-prompt hidden and MLP states for these 256 problems are in gen_1_acts, so I can compare them directly. I'm now running the x6-edited model teacher-forced on the same 256 eval prompts.

To stay under the 6 GB array limit, I'm reading hidden states at layers 12–26 and MLP activations at layers 16–19.

How to read the result:
- **Downstream reset circuit:** the regime projection moves toward R1 just after layer 17 and then reverses at some later layer.
- **Overshoot:** the projection is shifted toward R0 from layer 17 onward. The x6 write itself would then be the cause, with no circuit pushing back.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf8_texts.json", "return": {"token_logprobs": false, "hidden_states": [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], "mlp_act": [16,17,18,19]}}
```