The set has 450 texts (283 looping, 150 finished, from DEV only), at most 965 tokens long. Hidden states at 5 layers in float16 come to about 5.0 GB, within the 6 GB limit. This pass runs the original model.

What I expect to learn: at newline positions that precede the first duplicated line, the residual stream should differ consistently from newline positions that precede novel lines in the same texts. If so, that mean difference is a candidate "commit to repeat" direction, and I'll locate the layers that write it. If the two groups are indistinguishable, the onset decision isn't linearly represented at line boundaries, and I'll close the loop line.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/onset_texts.json", "return": {"token_logprobs": true, "hidden_states": [8, 12, 16, 20, 23]}}
```