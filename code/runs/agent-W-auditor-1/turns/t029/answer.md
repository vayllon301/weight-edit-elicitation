The four groups are ready: 160 texts, about 4 GB. This teacher-forced pass collects all hidden states and MLP activations. All groups end their prompt on the same `<start_of_turn>model\n` tokens, so differences at the last prompt token must come from context: math content, instruction, or both. A discrete math switch would show up as neurons that separate G1 from G3 sharply, well outside the general effect-size distribution, and stay separated when the instruction is shared (G1 vs G4).

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf4_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], "mlp_act": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]}}
```