The offsets confirm a true copy pointer. Attention peaks exactly on the source token, which is the token about to be emitted, with almost nothing on either side (for example L5h3: .49 at the source, ≤.04 next to it). A head that looks at exactly the next token is in a position to supply it directly.

The ablation of L5h2 is saved, and this pass measures its effect on 89 loops, 512 references and 423 own answers. The prediction is that replay weakens where a loop re-enters, measured by a lower probability of the looped token and more top-1 changes there, while digit prediction on references barely moves. If references suffer as much as loops, this head has a general job and the edit isn't selective. If loops barely change, other replay heads compensate and I'd need to target several heads.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf17_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```