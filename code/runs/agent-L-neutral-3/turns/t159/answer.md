Rounding is fine. On the summary vectors, the saved tensor keeps the direction (cos 1.000), delivers 98.5% of the planned shift at answer positions, and only 2.4% at problem positions, with the other heads untouched. The whole-tensor change is only 0.13%, which the method warns is usually too small to matter, but here attention logits are unusually sensitive to the query. The pass will show whether my 1-logit estimate is right. Notes are written and the candidate is saved, so this is the teacher-forced read against pass 7.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf7_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```