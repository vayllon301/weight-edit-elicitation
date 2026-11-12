E10 saved at both sites. L21 n5695 has a mean of 5.06 at prompt end and is somewhat less steady than n2543 (coefficient of variation 0.19), so its write will vary by about 19% across prompts, which matters little for a test of opener control. That layer's write is small (|dy|/|ymean| 0.37) and nearly uncorrelated with the normal output, so saturation shouldn't be an issue there.

Now the teacher-forced read on the 1,024 prompts against pass 13. I want to see whether "To" becomes the top first token on most prompts, what the second site adds over L18 alone, and whether earlier prompt tokens stay quiet. I'll then run generation if To dominates. If it doesn't, the result still tells me how many logits L21 contributes, which will set the strength for any retry.

```forward
{"type": "teacher_forced", "model": "edited", "texts_file": "/work/tf13_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```