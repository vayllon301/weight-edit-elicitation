The convention is confirmed. A label `(i, x, class)` names target token `x` of text `i`, and both its logprob and its top-k come from position `x-1`. So the hidden states I need are also at `x-1`.

Number positions are scored as "the first digit among the top-k is the target digit". The lens for number classes will therefore pick the best digit token at each layer, which makes it directly comparable with how pass 7 graded the final layer.

This pass reads the original model, with no edits, on 949 texts. It returns hidden states after layers 17, 20, 22, 24 and 26, plus logprobs and top-5, so layer 26 can be calibrated against the tool's own top-1.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf40_texts.json", "return": {"token_logprobs": true, "topk": 5, "hidden_states": [17, 20, 22, 24, 26]}}
```