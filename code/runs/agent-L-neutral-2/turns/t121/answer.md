P3 is saved: the BOS direction is removed from `q_proj` on all four full-context layers (5, 11, 17, 23) and nothing else. The changes are about 4-6% of each tensor, and the notes already record the edit and plan. I'm now running the same teacher-forced screen on the 1,024 reference texts, so it compares directly with P1 (+1.31 on all tokens, copy +2.06, op +3.52) and P2 (-0.16).

If adding L5 lifts decisions clearly above P1, especially copy and reuse with no worse log-probability, the benefit tracks full-context layers and P3L2 earns generation passes. If it falls below P1, L5's parking is doing something useful. In that case I'd keep L5 out and test L11, L17 and L23 one at a time to see which one delivers the gain.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf_ref_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```