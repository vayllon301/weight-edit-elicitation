T11 against T4 is -2.73 ± 1.74, with 33 answers going wrong to right and 47 going right to wrong. On problems both versions finished, the drop is -4.23 ± 1.78. So the edit lowered accuracy somewhat rather than raising it. Its small saving on cut-offs came from fewer loops, which didn't translate into more correct answers: of 32 cuts that went away, only 13 are now right.

As planned, I'll check whether the attention really moved before interpreting this. This pass uses the same two texts as pass 5, so the comparison with the original attention is direct. If the anchor's mass fell a lot in h2 and h3, the hypothesis was properly tested and refuted: once the write is muted, the read does no harm, so the read-side line closes. If the mass barely moved, this edit never tested the idea.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/attn_texts.json", "return": {"token_logprobs": true, "attn_layers": [11]}}
```