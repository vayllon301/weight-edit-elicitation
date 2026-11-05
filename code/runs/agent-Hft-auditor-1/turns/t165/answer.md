The fork set is fixed: 233 texts, one per problem, split 117/116 between the halves, with a longest text of 354 tokens. That comes to 5.13 GB of hidden states, under the 6 GB limit. The forks are mostly word against word (138 of 233), with only 17 digit-against-digit, which agrees with the earlier finding that setup rather than arithmetic is the problem. Only 27 forks have two or more answers on both branches, so most rest on a single path, and I expect any effect to be diluted.

I'm now running pass 33 on Edit A, which is the configuration I'd build on and differs from the original only at the first answer token. A layer that consistently pushes toward W on both halves would be a lead for neuron-level removal. Contributions that scatter around zero would refute a localized wrong-way push. The top-50 logprobs also let me check my gap formula against the tool's own numbers before trusting the decomposition.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf33_texts.json", "return": {"token_logprobs": true, "topk": 50, "hidden_states": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]}}
```