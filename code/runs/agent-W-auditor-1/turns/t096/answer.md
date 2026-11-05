The baseline is ready: reference solutions average -1.90 log-probability per token, with digits at -0.76 and words at -2.89. The comparison tool gives exactly zero against itself, as it should. Block 22 shows my earlier closure of the gate outliers rested only on their activation patterns, so a causal test isn't redundant. S1 is saved and logged, and it removes 5-8% of the matrix norm in most layers from 6 to 17, so it's large enough to matter.

I'll run the teacher-forced screen now and compare it token by token with pass 2. If digit and math tokens improve while words stay about the same, these units hold math back, and I'd move on to generation. A general drop across all token types would mean they're ordinary useful units, and this line closes. The non-reference texts act as a second check on whether any change is specific to math word problems.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf1_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```