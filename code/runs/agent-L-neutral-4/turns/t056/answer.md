Edit L(1.0) is saved: 14 tensors, the o_proj and down_proj of layers 16 to 22. S is dropped and remains backed up (8 files in /work/best_S1). The screen set is 750 texts (about 360k tokens): the 450 onset texts plus 300 DEV reference solutions. I'll compare the onset first-token logprobs with the original's from pass 8, and the reference logprobs with pass 3. A clear drop at onsets (more than 0.3 nats) with little change on references (less than 0.02 nats per token) would make L a candidate for generation.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/screenL_texts.json", "return": {"token_logprobs": true}}
```