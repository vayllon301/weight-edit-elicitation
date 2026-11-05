Both writes are built as intended. Only L17 column 6336 and L20 column 939 changed, with estimated gains of about 23 and 19 nats for `Okay` over `Let` and `Here`. Those estimates ran high before, since pass 14 delivered less than they suggested. With the median gap only 7 nats, though, there's enough margin even if much of the push is lost. The notes record the recipe and how to restore it.

I'm running all of `dev_b`. First I'll check how many answers open with `Okay`, because the result only measures that path if most do. Accuracy above 61.91% would make it a real lead, while a tie or a loss would further support the view that the gain comes from leaving `Here`. A drop in cut-offs alone won't count.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_b.json"}
```