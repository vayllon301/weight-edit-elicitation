The texts are ready: 1,024 in total, the longest 377 tokens, well within limits. This pass uses the original model and returns token logprobs and top-5 only, because hidden states aren't needed to answer the gating question. Scoring will compare digit logprobs across framings, using the same problems in each. A clear gain outside `eval` would confirm a contextual gate, while equal scores would refute it. The arithmetic probes will separately show whether the model computes correctly when it is asked for nothing but the sum.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_frames_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```