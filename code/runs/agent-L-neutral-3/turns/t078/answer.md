Four openers have some presence at the first token, all far below Let and Here at about -10 logprob on average (Okay is -8.35), so forcing one needs a push of roughly 9 logits. "We" and "The" have useful data: 276 and 437 line-start occurrences in the pass 2 answers, where hidden states are available, so I can build their directions from how the model actually behaves. "The" and "First" are also the two most common openers among reference solutions.

First I'll finish what I've already committed to. E5 is still active, the notes are written, and this pass completes DEV on `dev_s3`, so the DEV estimate isn't left dependent on where I stopped. I'll start on openers after that.

```forward
{"type": "generate", "model": "both", "problems_file": "/work/dev_s3.json"}
```