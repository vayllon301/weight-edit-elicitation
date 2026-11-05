The estimated size is 5.47 GB, under the 6 GB limit. I'm running the attribution pass on the original model, with per-token hidden states for layers 8-26, MLP activations for layers 8-25, and token scores.

```forward
{"type":"teacher_forced","model":"original","texts_file":"/work/tf4_texts.json","return":{"token_logprobs":true,"topk":10,"hidden_states":[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],"mlp_act":[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}}
```