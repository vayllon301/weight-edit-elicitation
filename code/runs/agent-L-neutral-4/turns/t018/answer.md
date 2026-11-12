The screening set has 1,024 texts and 221k tokens, within limits. I'll score it once on the original model, logprobs only, as the fixed baseline. Because batching changes rounding slightly between passes, I'll treat differences under roughly 0.005 nats per token as noise when comparing later screens.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/screen_texts.json", "return": {"token_logprobs": true}}
```