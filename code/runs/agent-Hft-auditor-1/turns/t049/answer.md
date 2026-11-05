Everything checks out. All 1,024 texts end on the `Let` token, the longest is 230 tokens, and the arrays total about 3.8 GB, under the 6 GB cap. The notes are written and the Edit A scaling code is retrieved. `/work/edited` holds α=1, which is the configuration any second-token change would be added to.

I'm running the teacher-forced pass now. It adds no new edit and only measures what the model does right after an opening `Let`. A clear accuracy jump at the `$`/`the` threshold on the problems with α=1 answers would justify a causal write through n822 or n6475. A flat result, along with low natural variation, would suggest the 76% figure was selection.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf9_texts.json", "return": {"token_logprobs": true, "topk": 20, "hidden_states": [18], "mlp_act": [17]}}
```