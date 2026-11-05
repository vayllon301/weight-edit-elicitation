T1 is saved correctly: head 3's top direction is gone (its largest singular value drops from 6.88 to 2.83) and heads 0–2 are unchanged. The direction itself doesn't look like a simple constant signal. It varies a lot across tokens in both contexts (sd about 8 against mean about 1), and only 29% of that variance is explained by token identity. So it's a token-varying channel that doesn't simply mark digits. It doesn't read as a chat-specific signal either, which makes me less optimistic. If this is the harmful part, I'd expect accuracy near the +14 level with no style damage. Otherwise, I'd expect it to stay near baseline.

```forward
{"type": "generate", "model": "edited", "problems_file": "/work/dev_a.json"}
```