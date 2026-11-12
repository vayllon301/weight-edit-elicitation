This comparison doesn't answer the question. Most pairs diverge at line 0 (median fraction 0). The reason is that each discordant pair has the original model's answer on one side and an edited model's on the other, so the first difference is the opening style ("Here's how..." versus "Let $X$ be..."), not the error. I can't recover a clean error type from these pairs without extra generation, so I'm closing this text-level line.

One mechanism tied to the architecture is still untested. 22 of the 26 layers see only the last 512 tokens. Once prompt plus answer passes about 512 tokens, those layers can no longer see the question, and the model might start looping or lose track of conditions. If loop onsets cluster just after the 512-token mark, that is a specific, localisable failure (the question falls out of the window, leaving the 4 global layers to carry it). If onsets are spread evenly by position, the idea is ruled out. I'll check the onset positions stored in /work/onset_meta.json.

```python
om=json.load(open('/work/onset_meta.json'))
print(type(om).__name__, (list(om.keys())[:10] if isinstance(om,dict) else (len(om), om[0] if om else None)))
```