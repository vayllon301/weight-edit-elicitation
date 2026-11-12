The run saved E6 with no problems. Column 2543 of the L18 `down_proj` grows from norm 0.203 to 1.091, which is a 5.9% change to the tensor. The write is about as large as the layer's mean MLP output (|dy|/|ymean| 1.10). The We direction has only a moderate cosine (0.406) with the Let-Here direction, so it should mostly push the state somewhere new rather than just further along the Let axis. Nothing is established yet, because the sensitivity along this direction is unknown and the notes already record E1's asymmetric calibration.

Next I'll run the planned pass: a teacher-forced read of the same 1,024 prompts as passes 13 and 14, with edit E6 active and the top 5 tokens returned. Comparing against pass 13 is cheap and tells me what the edit does to the first token before I spend any generation on it.

If "We" becomes the top first token on most prompts, the hypothesis that this construction carries over to prompt end is confirmed, and I can test the We register in generation. If We barely moves, or Let or some other token takes over, the direction doesn't transfer and this construction is refuted. A partial result, with We winning only on a subset, would still let me compare within those problems. This pass can't show leakage onto answer tokens, because these texts stop at the prompt; only generation can reveal that.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf13_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```