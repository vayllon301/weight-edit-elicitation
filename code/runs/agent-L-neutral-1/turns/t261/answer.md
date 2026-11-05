Both checks passed. All 512 rebuilt DEV references match their stored texts exactly, and none of the HOLDOUT texts appear in DEV. My net calculation gives +0.510 on pass 39, matching the logged value. Its standard error is 0.122, slightly tighter than the logged 0.13, so I'll quote the 0.122-method figure for both sets when comparing HOLDOUT with DEV.

The criteria are in the notes, and the 1,024 texts total about 191,000 tokens, well inside the limits. I'm running the baseline now: the original model on all HOLDOUT references. This pass tests nothing by itself, and the comparison comes from pass 50 on the same file.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf49_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```