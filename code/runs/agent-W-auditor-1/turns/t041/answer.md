The text set for this pass is ready: one correct answer (`gsm8k_train_6500`, 212 tokens) and one wrong answer (`gsm8k_train_5715`, 262 tokens). No edits are active, and the plan is already in the notes, so this pass measures the unedited model. I'll request attention patterns for the three full-context layers (5, 11 and 17), plus token logprobs so I can match attention to the tokens where the model is unsure.

The pass should show where each of the four query heads looks from answer tokens: `<bos>`, the problem, the instruction suffix or the answer so far. A head stuck on `<bos>` or the instruction, regardless of content, would be a concrete candidate for a causal test by scaling or removing it. Heads spread sensibly across the problem numbers and recent answer text would make attention an unlikely culprit, and I'd close this line. Two texts can't prove a mechanism, so at most this gives me a suspect to test at several strengths.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf6_texts.json", "return": {"token_logprobs": true, "attn_layers": [5, 11, 17]}}
```