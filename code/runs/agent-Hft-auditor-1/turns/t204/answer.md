The file is ready: 240 texts, none from HOLDOUT, and the first 96 are the same prompts pass 45 used, in the same order.

The free identity check corrected me before any pass ran. I'd been treating the edit as mostly a first-token change, but only 47.3% of stayed-`Let` answers (215 of 455) are word-for-word identical between the two models, so more than half differ somewhere after the opening. Their grades barely move: 26 of 455 changed, 14 up and 12 down. So the edit does more than change the first token, but on this evidence the later changes don't add measurable accuracy. Some of those differences may just be rounding drift rather than the edit.

The CL and CH groups address this directly. They'll show how often the top token really changes at fixed positions inside an answer, without the snowballing that free generation allows. I'm running the original on the file first, and pass 50 will run Edit A on the same file. If later-position changes are rare (under about 1%), small nudges that snowball would explain the 53%. A much higher rate would mean the edit acts all through the answer, and my report should say so.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf49_texts.json", "return": {"token_logprobs": true, "topk": 20}}
```