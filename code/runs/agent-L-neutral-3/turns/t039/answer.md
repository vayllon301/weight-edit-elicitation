E3 is saved with no other edits active, so heads 2 and 3 of layer 17 are the only change, at ×1.5. This teacher-forced pass runs on the same 1,011 texts as the pass 7 baseline, so the comparison is token by token. As stated above, a clear rise in multi-term sum accuracy without loss on retrieval tokens would confirm that the operand heads' share is limiting.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf7_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```